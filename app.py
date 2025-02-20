# app_new.py
import io
import numpy as np
import streamlit as st
import cv2
from PIL import Image, ExifTags

# ------------------------------
# Utility Functions
# ------------------------------
def load_image(file, grayscale=True):
    """Load image from uploaded file; return OpenCV float32 image and raw bytes."""
    data = file.getvalue()
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    if img is None:
        st.error(f"⚠️ Could not load image: {file.name}")
    return img.astype(np.float32), data

def adjust_exposure(image_bytes):
    """Extract exposure time from image EXIF data; default to 1.0 if missing."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "ExposureTime":
                    if isinstance(value, tuple) and value[1] != 0:
                        return value[0] / value[1]
                    else:
                        return float(value)
        return 1.0
    except Exception:
        return 1.0

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """Enhance image contrast using CLAHE and normalize to [0,1]."""
    norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(norm_img)
    return clahe_img.astype(np.float32) / 255.0

def auto_roi_mask(ref_norm):
    """
    Automatically compute a binary ROI mask from a normalized reference image.
    Uses a Gaussian blur and Otsu thresholding.
    """
    img_uint8 = (ref_norm * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def compute_bbox_from_mask(mask):
    """Compute bounding box (ymin, ymax, xmin, xmax) from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("No region found in ROI!")
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

def get_exif_data(file_obj):
    """Extract and return EXIF data as a dict from a PIL image."""
    try:
        pil_img = Image.open(file_obj)
        exif_data = pil_img._getexif()
        if exif_data:
            exif = {}
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif[decoded] = value
            return exif
        return {}
    except Exception:
        return {}

def image_to_bytes(image):
    """Convert an image (values in [0,1]) to PNG bytes."""
    success, buf = cv2.imencode('.png', (np.clip(image, 0, 1)*255).astype(np.uint8))
    return buf.tobytes()

def safe_display(image):
    """Clip image values to [0,1] for safe display."""
    return np.clip(image, 0.0, 1.0)

def format_file_stats(uploaded_file):
    """Return a formatted string with file stats and some EXIF info."""
    exif = get_exif_data(uploaded_file)
    size_kb = len(uploaded_file.getvalue()) / 1024
    info = f"**Filename:** {uploaded_file.name}\n\n"
    info += f"**Size:** {size_kb:.1f} KB\n\n"
    if "ExposureTime" in exif:
        info += f"**ExposureTime:** {exif['ExposureTime']}\n\n"
    else:
        info += "**ExposureTime:** Not available\n\n"
    return info

def load_example_file(path, default_name):
    """Load a file from the assets folder as a BytesIO with a name attribute."""
    with open(path, "rb") as f:
        data = f.read()
    file_obj = io.BytesIO(data)
    file_obj.name = default_name
    return file_obj

# ------------------------------
# Processing Functions
# ------------------------------
def process_uv_image(uv_file, bbox, ref_crop, params):
    """
    Process a single UV image:
      - Load the image and correct for exposure.
      - Enhance the image with CLAHE.
      - Crop to ROI and compute the absolute difference with the reference crop.
      - From the difference compute:
          a) a binary mask via Otsu thresholding.
          b) a continuous (gradient) mask scaled and normalized.
    """
    uv_img, uv_data = load_image(uv_file, grayscale=True)
    exposure = adjust_exposure(uv_data)
    uv_corrected = uv_img / (exposure if exposure != 0 else 1.0)
    uv_norm = apply_clahe(uv_corrected, 
                          clip_limit=params['clahe_clip'], 
                          tile_grid_size=(params['clahe_tile'], params['clahe_tile']))
    uv_crop = uv_norm[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    
    # Compute the absolute difference between UV and reference crops
    diff = np.abs(uv_crop - ref_crop)
    
    # Binary Mask: threshold difference using Otsu
    diff_uint8 = (diff * 255).astype(np.uint8)
    _, binary = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = (binary > 0).astype(np.float32)
    
    # Continuous (gradient) mask: scale and clip
    scaled_diff = diff * params['diff_scale']
    cont_mask = np.clip(scaled_diff, 0, 1)
    
    return uv_crop, diff, binary_mask, cont_mask, exposure

def process_pipeline_super(ref_file, mask_file, uv_files, params):
    """
    Super processing pipeline that computes both binary and gradient masks.
    
    Steps:
      1. Load and enhance the reference image.
      2. Determine ROI (using user-supplied mask or auto-computed mask).
      3. Crop the reference image to ROI.
      4. For each UV image:
           - Correct for exposure, enhance, crop.
           - Compute difference with reference crop.
           - Generate a binary mask (via Otsu) and a continuous gradient mask.
      5. Fuse UV outputs:
           - Combine binary masks (pixel-wise maximum) and refine with morphological operations.
           - Average continuous masks and smooth with Gaussian blur.
      6. Generate full-frame masks by placing the ROI output back into the reference frame.
    """
    results = {}
    
    # Process Reference Image
    ref_img, ref_data = load_image(ref_file, grayscale=True)
    ref_norm = apply_clahe(ref_img, clip_limit=params['clahe_clip'],
                           tile_grid_size=(params['clahe_tile'], params['clahe_tile']))
    results['Reference'] = ref_norm
    
    # Determine ROI mask: if user provides a mask, use it; otherwise auto-compute.
    if mask_file:
        mask_img, _ = load_image(mask_file, grayscale=True)
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    else:
        mask_bin = auto_roi_mask(ref_norm)
    results['ROI_Mask'] = mask_bin
    
    # Compute bounding box from ROI
    try:
        bbox = compute_bbox_from_mask(mask_bin)
    except Exception as e:
        st.error("Error computing ROI: " + str(e))
        return None
    results['BoundingBox'] = bbox
    
    # Crop the reference image to the ROI
    ref_crop = ref_norm[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    results['CroppedReference'] = ref_crop
    
    # Process each UV image
    uv_details = []
    combined_binary = None
    combined_gradient = None
    for uv in uv_files:
        try:
            uv_crop, diff, bin_mask, cont_mask, exposure = process_uv_image(uv, bbox, ref_crop, params)
            uv_details.append({
                'name': uv.name,
                'CroppedUV': uv_crop,
                'Difference': diff,
                'BinaryMask': bin_mask,
                'ContinuousMask': cont_mask,
                'Exposure': exposure,
                'EXIF': get_exif_data(uv)
            })
            # Fuse binary masks via pixel-wise maximum
            if combined_binary is None:
                combined_binary = bin_mask
            else:
                combined_binary = np.maximum(combined_binary, bin_mask)
            # Fuse gradient masks via averaging
            if combined_gradient is None:
                combined_gradient = cont_mask
            else:
                combined_gradient = (combined_gradient + cont_mask) / 2.0
        except Exception as e:
            st.error(f"Error processing {uv.name}: {e}")
    results['UV_Details'] = uv_details
    
    if combined_binary is None or combined_gradient is None:
        st.error("No valid UV images processed.")
        return None

    # Refine the combined binary mask with morphological operations
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph_kernel'], params['morph_kernel']))
    refined_binary = cv2.morphologyEx((combined_binary * 255).astype(np.uint8), cv2.MORPH_CLOSE, morph_kernel, iterations=params['morph_iter'])
    refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_OPEN, morph_kernel, iterations=params['morph_iter'])
    refined_binary = refined_binary.astype(np.float32) / 255.0
    results['FinalBinaryCropped'] = refined_binary
    
    # Create full-frame binary mask
    full_binary = np.zeros_like(ref_norm, dtype=np.float32)
    full_binary[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1] = refined_binary
    results['FinalBinaryFull'] = full_binary
    
    # Smooth the combined gradient mask with Gaussian blur
    kernel = (params['final_blur_kernel'], params['final_blur_kernel'])
    smooth_gradient = cv2.GaussianBlur(combined_gradient, kernel, params['final_blur_sigma'])
    smooth_gradient = np.clip(smooth_gradient, 0, 1)
    results['FinalGradientCropped'] = smooth_gradient
    
    # Create full-frame gradient mask
    full_gradient = np.zeros_like(ref_norm, dtype=np.float32)
    full_gradient[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1] = smooth_gradient
    results['FinalGradientFull'] = full_gradient
    
    return results

# ------------------------------
# Main UI: Streamlit App Layout
# ------------------------------
st.set_page_config(page_title="Super UV Mask Maker", layout="wide")
st.title("Super UV Mask Maker 🚀")
st.write("Generate ultra-precise UV masks for resin 3D printing by combining binary and gradient approaches. Adjust parameters to fine-tune every step.")

# Session state for file uploads
if "ref_upload" not in st.session_state:
    st.session_state.ref_upload = None
if "mask_upload" not in st.session_state:
    st.session_state.mask_upload = None
if "uv_uploads" not in st.session_state:
    st.session_state.uv_uploads = []

st.sidebar.header("Upload Your Images")
uploaded_ref = st.sidebar.file_uploader("Reference Image", type=["png", "jpg", "jpeg"], key="ref_upload_input")
uploaded_mask = st.sidebar.file_uploader("Mask Image (Optional)", type=["png", "jpg", "jpeg"], key="mask_upload_input")
uploaded_uv = st.sidebar.file_uploader("UV Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="uv_uploads_input")

# Update session state if new files are provided
if uploaded_ref is not None:
    st.session_state.ref_upload = uploaded_ref
if uploaded_mask is not None:
    st.session_state.mask_upload = uploaded_mask
if uploaded_uv is not None and len(uploaded_uv) > 0:
    st.session_state.uv_uploads = uploaded_uv

# Button: Load example images from assets folder
if st.sidebar.button("Load Example Images"):
    try:
        st.session_state.ref_upload = load_example_file("assets/reference.png", "reference.png")
        st.session_state.mask_upload = load_example_file("assets/mask.png", "mask.png")
        st.session_state.uv_uploads = [load_example_file("assets/uv_0.004000s_358.png", "uv_sample.png")]
        st.sidebar.success("Example images loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading example images: {e}")

# Display file information on the sidebar
if st.session_state.ref_upload:
    st.sidebar.info(f"Reference: {st.session_state.ref_upload.name}")
if st.session_state.mask_upload:
    st.sidebar.info(f"Mask: {st.session_state.mask_upload.name}")
if st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0:
    uv_names = ", ".join([uv.name for uv in st.session_state.uv_uploads])
    st.sidebar.info(f"UV Images: {uv_names}")

# ------------------------------
# Sidebar: Adjustable Processing Parameters
# ------------------------------
st.sidebar.header("Processing Parameters")
clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, step=0.1)
clahe_tile = st.sidebar.slider("CLAHE Tile Grid Size", 4, 16, 8, step=1)
diff_scale = st.sidebar.slider("Difference Scale Factor", 0.5, 5.0, 1.0, step=0.1)
morph_kernel = st.sidebar.slider("Morphological Kernel Size (Odd)", 3, 21, 7, step=2)
morph_iter = st.sidebar.slider("Morphological Iterations", 1, 10, 2, step=1)
final_blur_kernel = st.sidebar.slider("Final Gaussian Blur Kernel (Odd)", 3, 31, 11, step=2)
final_blur_sigma = st.sidebar.slider("Final Gaussian Blur Sigma", 0.0, 10.0, 2.0, step=0.1)

params = {
    'clahe_clip': clahe_clip,
    'clahe_tile': clahe_tile,
    'diff_scale': diff_scale,
    'morph_kernel': morph_kernel,
    'morph_iter': morph_iter,
    'final_blur_kernel': final_blur_kernel,
    'final_blur_sigma': final_blur_sigma
}

# ------------------------------
# Quick Pipeline Preview
# ------------------------------
st.header("Pipeline Preview")
preview_cols = st.columns(3)
with preview_cols[0]:
    st.subheader("Reference")
    if st.session_state.ref_upload:
        st.image(Image.open(st.session_state.ref_upload), use_container_width=True)
        with st.expander("View Details"):
            st.markdown(format_file_stats(st.session_state.ref_upload))
    else:
        st.write("No Reference image loaded.")
with preview_cols[1]:
    st.subheader("Mask / ROI")
    if st.session_state.mask_upload:
        st.image(Image.open(st.session_state.mask_upload), use_container_width=True)
        with st.expander("View Details"):
            st.markdown(format_file_stats(st.session_state.mask_upload))
    else:
        st.write("No Mask provided; auto-ROI will be computed.")
with preview_cols[2]:
    st.subheader("UV")
    if st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0:
        st.image(Image.open(st.session_state.uv_uploads[0]), use_container_width=True)
        with st.expander("View Details"):
            st.markdown(format_file_stats(st.session_state.uv_uploads[0]))
    else:
        st.write("No UV image loaded.")

# ------------------------------
# Main Processing Button and Output
# ------------------------------
if st.sidebar.button("Process Images"):
    if not st.session_state.ref_upload or not (st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0):
        st.error("Please upload at least a Reference Image and one UV Image (or load example images).")
    else:
        with st.spinner("Processing images..."):
            output = process_pipeline_super(
                st.session_state.ref_upload,
                st.session_state.mask_upload,
                st.session_state.uv_uploads,
                params
            )
        if output:
            st.success("Processing complete!")
            
            # Display Intermediate Results
            st.header("Intermediate Results")
            col_ref, col_roi = st.columns(2)
            with col_ref:
                st.subheader("Enhanced Reference")
                st.image(safe_display(output['Reference']), use_container_width=True)
            with col_roi:
                st.subheader("ROI Mask")
                st.image(safe_display(output['ROI_Mask'] / 255.0), use_container_width=True)
            st.subheader("Bounding Box")
            st.write(f"Coordinates: {output['BoundingBox']}")
            st.subheader("Cropped Reference")
            st.image(safe_display(output['CroppedReference']), use_container_width=True)
            
            st.subheader("UV Details")
            for uv in output['UV_Details']:
                st.markdown(f"**{uv['name']}**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(safe_display(uv['CroppedUV']), caption="Cropped UV", use_container_width=True)
                with c2:
                    st.image(safe_display(uv['Difference']), caption="Difference Map", use_container_width=True)
                with c3:
                    st.image(safe_display(uv['BinaryMask']), caption="Binary Mask", use_container_width=True)
                with c4:
                    st.image(safe_display(uv['ContinuousMask']), caption="Gradient Mask", use_container_width=True)
                    st.caption(f"Exposure: {uv['Exposure']}")
            
            # Final Outputs: Binary Masks
            st.header("Final Binary Mask Outputs")
            col_bin1, col_bin2 = st.columns(2)
            with col_bin1:
                st.image(safe_display(output['FinalBinaryCropped']), caption="Cropped Binary Mask", use_container_width=True)
                st.download_button(
                    label="Download Cropped Binary Mask",
                    data=image_to_bytes(output['FinalBinaryCropped']),
                    file_name="final_cropped_binary_mask.png",
                    mime="image/png"
                )
            with col_bin2:
                st.image(safe_display(output['FinalBinaryFull']), caption="Full Binary Mask", use_container_width=True)
                st.download_button(
                    label="Download Full Binary Mask",
                    data=image_to_bytes(output['FinalBinaryFull']),
                    file_name="final_full_binary_mask.png",
                    mime="image/png"
                )
            
            # Final Outputs: Gradient Masks
            st.header("Final Gradient Mask Outputs")
            col_grad1, col_grad2 = st.columns(2)
            with col_grad1:
                st.image(safe_display(output['FinalGradientCropped']), caption="Cropped Gradient Mask", use_container_width=True)
                st.download_button(
                    label="Download Cropped Gradient Mask",
                    data=image_to_bytes(output['FinalGradientCropped']),
                    file_name="final_cropped_gradient_mask.png",
                    mime="image/png"
                )
            with col_grad2:
                st.image(safe_display(output['FinalGradientFull']), caption="Full Gradient Mask", use_container_width=True)
                st.download_button(
                    label="Download Full Gradient Mask",
                    data=image_to_bytes(output['FinalGradientFull']),
                    file_name="final_full_gradient_mask.png",
                    mime="image/png"
                )
            
            st.success("Your super UV masks are ready for precision 3D printing!")
