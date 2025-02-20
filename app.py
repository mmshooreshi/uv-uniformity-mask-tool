import streamlit as st
import cv2
import numpy as np
import re
import io

# ================================================
#         HELPER FUNCTIONS (Image I/O)
# ================================================
def load_grayscale_image_from_bytes(uploaded_file):
    """Load an image from a stream as grayscale (float32)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error(f"Failed to load image: {uploaded_file.name}")
    # Reset file pointer for future use if needed.
    uploaded_file.seek(0)
    return img.astype(np.float32)

def convert_cv2_to_bytes(image):
    """Convert a CV2 image (grayscale or BGR) to PNG bytes."""
    ret, buf = cv2.imencode('.png', image)
    return buf.tobytes()

# ================================================
#         PROCESSING HELPER FUNCTIONS
# ================================================
def get_exposure_time(filename):
    """Extract exposure time from filename (e.g., 'uv_0.5s.png')."""
    match = re.search(r'uv_([\d.]+)s', filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Exposure time not found in filename: {filename}")

def load_mask_and_bbox(mask_image):
    """
    Given a grayscale mask image (float32), threshold it,
    then compute and return the binary mask, bounding box, and boolean mask.
    """
    _, mask_bin = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    mask_bool = mask_bin > 0
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask has no non-zero pixels!")
    bbox = (ys.min(), ys.max(), xs.min(), xs.max())
    return mask_bool, bbox, mask_bin

def crop_to_bbox(image, bbox):
    """Crop an image to the bounding box (ymin, ymax, xmin, xmax)."""
    ymin, ymax, xmin, xmax = bbox
    return image[ymin:ymax+1, xmin:xmax+1]

def normalize_by_exposure(image, exposure_time):
    """Divide the image by its exposure time."""
    if exposure_time == 0:
        raise ValueError("Exposure time is zero!")
    return image / exposure_time

def normalize_intensity_map(image, epsilon=1e-6):
    """Normalize image intensities to the [0,1] range."""
    min_val, max_val = image.min(), image.max()
    if max_val - min_val < epsilon:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)

def refine_inverse_mask_super_smooth(mask, blur_iterations=50, kernel_size=51):
    """
    Refine the mask to be super smooth (with no visible contours)
    by applying a morphological closing followed by many iterations of Gaussian blur.
    """
    # Convert to 8-bit image.
    mask_uint8 = np.uint8(np.clip(mask * 255.0, 0, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    smoothed = closed.copy()
    for i in range(blur_iterations):
        smoothed = cv2.GaussianBlur(smoothed, (kernel_size, kernel_size), 0)
    refined = smoothed.astype(np.float32) / 255.0
    return normalize_intensity_map(refined)

# ================================================
#           PROCESSING PIPELINE FUNCTION
# ================================================
def process_pipeline(ref_image, mask_image, uv_files, params):
    results = {}  # Dictionary to hold all outputs

    # 1. Reference Image is already loaded.
    results['reference'] = ref_image

    # 2. Process Mask and extract bounding box.
    mask_bool, bbox, mask_bin = load_mask_and_bbox(mask_image)
    results['mask'] = mask_bin
    results['bbox'] = bbox

    # 3. Crop the reference image to the ROI defined by the mask.
    ref_cropped = crop_to_bbox(ref_image, bbox)
    results['cropped_reference'] = ref_cropped

    # 4. Process each UV image.
    diff_maps = []
    uv_details = []  # To store each UV's cropped and diff results.
    for uv_file in uv_files:
        try:
            uv_image = load_grayscale_image_from_bytes(uv_file)
            exposure_time = get_exposure_time(uv_file.name)
            norm_uv = normalize_by_exposure(uv_image, exposure_time)
            uv_cropped = crop_to_bbox(norm_uv, bbox)
            diff = cv2.absdiff(uv_cropped, ref_cropped)
            diff = diff * params['DIFF_SCALE_FACTOR']
            diff = normalize_intensity_map(diff, epsilon=params['EPSILON'])
            diff_maps.append(diff)
            uv_details.append({
                'name': uv_file.name,
                'uv_cropped': uv_cropped,
                'diff': diff
            })
        except Exception as e:
            st.error(f"Error processing {uv_file.name}: {e}")
    results['uv_details'] = uv_details

    if not diff_maps:
        st.error("No UV images processed.")
        return None

    # 5. Combine the difference maps (average).
    combined_diff = np.mean(np.stack(diff_maps), axis=0)
    results['combined_diff'] = combined_diff

    # 6. Create refined mask and its inverse.
    refined_mask = refine_inverse_mask_super_smooth(
        combined_diff,
        blur_iterations=params['blur_iterations'],
        kernel_size=params['kernel_size']
    )
    results['refined_mask'] = refined_mask
    inverse_mask_refined = (1.0 - refined_mask / 1.5) / 3
    results['inverse_mask_refined'] = inverse_mask_refined

    # 7. Create final output masks.
    cropped_mask = np.uint8(normalize_intensity_map(inverse_mask_refined) * 255)
    results['final_cropped_mask'] = cropped_mask

    full_mask = np.zeros_like(ref_image, dtype=np.float32)
    ymin, ymax, xmin, xmax = bbox
    full_mask[ymin:ymax+1, xmin:xmax+1] = inverse_mask_refined
    final_uncropped_mask = np.uint8(normalize_intensity_map(full_mask) * 255)
    results['final_uncropped_mask'] = final_uncropped_mask

    return results

# ================================================
#           STREAMLIT UI IMPLEMENTATION
# ================================================
st.set_page_config(page_title="UV Uniformity Tool", layout="wide")
st.title("UV Uniformity Full Detail Tool")
st.write("This tool processes your images to analyze UV uniformity with full control over each parameter and displays every intermediate step.")

# ---- Sidebar for file uploads and parameters ----
st.sidebar.header("Upload Your Files")
ref_file = st.sidebar.file_uploader("Reference Image (e.g., reference.png)", type=["png", "jpg", "jpeg"])
mask_file = st.sidebar.file_uploader("Mask Image (e.g., mask.png)", type=["png", "jpg", "jpeg"])
uv_files = st.sidebar.file_uploader("UV Images (e.g., uv_0.5s.png)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.sidebar.header("Parameters")
diff_scale = st.sidebar.slider("DIFF_SCALE_FACTOR", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
epsilon_val = st.sidebar.number_input("EPSILON", value=1e-6, format="%.7f")
blur_iterations = st.sidebar.slider("Blur Iterations", min_value=1, max_value=100, value=30, step=1)
kernel_size = st.sidebar.slider("Kernel Size (odd numbers)", min_value=3, max_value=101, value=41, step=2)
save_intermediate = st.sidebar.checkbox("Show Intermediate Steps", value=True)

params = {
    'DIFF_SCALE_FACTOR': diff_scale,
    'EPSILON': epsilon_val,
    'blur_iterations': blur_iterations,
    'kernel_size': kernel_size,
    'SAVE_INTERMEDIATE': save_intermediate
}

# ---- Process Button ----
if st.sidebar.button("Process Images"):
    if not ref_file or not mask_file or not uv_files:
        st.error("Please upload all required images: Reference, Mask, and at least one UV image.")
    else:
        with st.spinner("Processing..."):
            try:
                # Load reference and mask images.
                ref_image = load_grayscale_image_from_bytes(ref_file)
                mask_image = load_grayscale_image_from_bytes(mask_file)
                
                # Run processing pipeline.
                results = process_pipeline(ref_image, mask_image, uv_files, params)
                if results is None:
                    st.error("Processing failed.")
                else:
                    st.success("Processing complete!")
                    
                    # --------------------------
                    # Display Intermediate Results
                    # --------------------------
                    st.header("Intermediate Steps")
                    
                    with st.expander("1. Reference Image"):
                        st.image(normalize_intensity_map(results['reference']), caption="Reference Image", use_column_width=True)
                    
                    with st.expander("2. Mask and Bounding Box"):
                        st.image(results['mask'], caption="Binary Mask", use_column_width=True)
                        bbox = results['bbox']
                        st.write(f"Bounding Box: (ymin: {bbox[0]}, ymax: {bbox[1]}, xmin: {bbox[2]}, xmax: {bbox[3]})")
                    
                    with st.expander("3. Cropped Reference Image"):
                        st.image(normalize_intensity_map(results['cropped_reference']), caption="Cropped Reference", use_column_width=True)
                    
                    with st.expander("4. UV Images & Their Difference Maps"):
                        for uv in results['uv_details']:
                            st.subheader(uv['name'])
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(normalize_intensity_map(uv['uv_cropped']), caption="Cropped UV", use_column_width=True)
                            with col2:
                                st.image(uv['diff'], caption="Difference Map", use_column_width=True)
                    
                    with st.expander("5. Combined Difference Map"):
                        st.image(normalize_intensity_map(results['combined_diff']), caption="Combined Difference", use_column_width=True)
                    
                    with st.expander("6. Refined Mask and Inverse Mask"):
                        st.image(results['refined_mask'], caption="Refined Mask", use_column_width=True)
                        st.image(normalize_intensity_map(results['inverse_mask_refined']), caption="Inverse Refined Mask", use_column_width=True)
                    
                    # --------------------------
                    # Display Final Outputs
                    # --------------------------
                    st.header("Final Outputs")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(results['final_cropped_mask'], caption="Final Cropped Mask", use_column_width=True)
                        png_bytes = convert_cv2_to_bytes(results['final_cropped_mask'])
                        st.download_button(label="Download Cropped Mask", data=png_bytes, file_name="output_mask_cropped.png", mime="image/png")
                    with col2:
                        st.image(results['final_uncropped_mask'], caption="Final Uncropped Mask", use_column_width=True)
                        png_bytes2 = convert_cv2_to_bytes(results['final_uncropped_mask'])
                        st.download_button(label="Download Uncropped Mask", data=png_bytes2, file_name="output_mask_uncropped.png", mime="image/png")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
