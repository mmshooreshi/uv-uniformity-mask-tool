import io
import numpy as np
import streamlit as st
import cv2
from PIL import Image, ExifTags

# ------------------------------
# Utility Functions
# ------------------------------
def load_image_and_data(uploaded_file, grayscale=True):
    """Load image from uploaded file; return OpenCV float32 image and its raw bytes."""
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    if img is None:
        st.error(f"⚠️ Could not load image: {uploaded_file.name}")
    return img.astype(np.float32), data

def extract_exposure(image_bytes):
    """Extract exposure time from image metadata via PIL; default to 1.0 if missing."""
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

def image_to_bytes(image):
    """Convert an OpenCV image to PNG bytes for downloads."""
    success, buf = cv2.imencode('.png', image)
    return buf.tobytes()

def safe_display(image):
    """Clip image array to [0,1] for safe display in Streamlit."""
    return np.clip(image, 0.0, 1.0)

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
    # Include other EXIF fields as desired...
    return info

# ------------------------------
# Processing Helper Functions
# ------------------------------
def create_default_mask(shape):
    """Generate a full-white mask for given dimensions."""
    return np.ones(shape, dtype=np.uint8) * 255

def compute_bbox(mask):
    """Compute bounding box (ymin, ymax, xmin, xmax) from a binary mask."""
    mask_bool = mask > 0
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("No non-zero pixels found in mask!")
    return (ys.min(), ys.max(), xs.min(), xs.max())

def crop_image(image, bbox):
    """Crop an image using the bounding box coordinates."""
    ymin, ymax, xmin, xmax = bbox
    return image[ymin:ymax+1, xmin:xmax+1]

def normalize(image, epsilon=1e-6):
    """Normalize image intensities to the [0,1] range."""
    min_val, max_val = image.min(), image.max()
    if max_val - min_val < epsilon:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)

def smooth_mask(diff_img, iterations=30, kernel_size=41):
    """Smooth the difference image to generate a refined mask."""
    mask_8bit = np.uint8(np.clip(diff_img * 255.0, 0, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_8bit, cv2.MORPH_CLOSE, kernel)
    smooth = closed.copy()
    for _ in range(iterations):
        smooth = cv2.GaussianBlur(smooth, (kernel_size, kernel_size), 0)
    refined = smooth.astype(np.float32) / 255.0
    return normalize(refined)

# ------------------------------
# Processing Pipeline
# ------------------------------
def process_pipeline(ref_file, mask_file, uv_files, params):
    results = {}

    # Load Reference Image
    ref_img, ref_data = load_image_and_data(ref_file, grayscale=True)
    results['Reference'] = normalize(ref_img)

    # Load Mask Image (or create default if missing)
    if mask_file:
        mask_img, _ = load_image_and_data(mask_file, grayscale=True)
    else:
        mask_img = create_default_mask(ref_img.shape)
    results['Mask'] = mask_img

    # Compute bounding box and crop reference image
    try:
        bbox = compute_bbox(mask_img)
    except Exception as e:
        st.error("Error computing bounding box: " + str(e))
        return None
    results['BoundingBox'] = bbox
    ref_crop = crop_image(ref_img, bbox)
    results['CroppedReference'] = normalize(ref_crop)

    # Process each UV image
    uv_diffs = []
    uv_details = []
    for uv in uv_files:
        try:
            uv_img, uv_data = load_image_and_data(uv, grayscale=True)
            exposure = extract_exposure(uv_data)
            norm_uv = uv_img / exposure if exposure != 0 else uv_img
            uv_crop = crop_image(norm_uv, bbox)
            diff = cv2.absdiff(uv_crop, ref_crop)
            diff = diff * params['scale']
            diff = normalize(diff, epsilon=params['epsilon'])
            uv_diffs.append(diff)
            uv_details.append({
                'name': uv.name,
                'CroppedUV': normalize(uv_crop),
                'Difference': diff,
                'Exposure': exposure,
                'EXIF': get_exif_data(uv)
            })
        except Exception as e:
            st.error(f"Error processing {uv.name}: {e}")
    results['UV_Details'] = uv_details

    if not uv_diffs:
        st.error("No valid UV images processed.")
        return None

    # Combine differences and generate refined masks
    combined_diff = np.mean(np.stack(uv_diffs), axis=0)
    results['CombinedDifference'] = normalize(combined_diff)
    refined = smooth_mask(combined_diff, iterations=params['iterations'], kernel_size=params['ksize'])
    results['RefinedMask'] = refined
    inverse_mask = (1.0 - refined / 1.5) / 3
    results['InverseMask'] = normalize(inverse_mask)

    # Create final output masks
    final_cropped = np.uint8(normalize(inverse_mask) * 255)
    results['FinalCroppedMask'] = final_cropped
    full_mask = np.zeros_like(ref_img, dtype=np.float32)
    ymin, ymax, xmin, xmax = bbox
    full_mask[ymin:ymax+1, xmin:xmax+1] = inverse_mask
    final_full = np.uint8(normalize(full_mask) * 255)
    results['FinalFullMask'] = final_full

    return results

# ------------------------------
# Sidebar: File Uploads & Example Loader
# ------------------------------
def load_example_file(path, default_name):
    """Load a file from the assets folder as a BytesIO with a name attribute."""
    with open(path, "rb") as f:
        data = f.read()
    file_obj = io.BytesIO(data)
    file_obj.name = default_name
    return file_obj


# ------------------------------
# Main UI: Title and Description
# ------------------------------
st.set_page_config(page_title="Aggressive UV Mask Maker", layout="wide")
st.title("Aggressive UV Mask Maker 🚀")
st.write("Generate ultra-customized mask images for UV exposure in resin 3D printing. Adjust parameters, inspect every step, and download your outputs.")

# Use session state to store file uploads
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

# Update session state from uploader if new files are provided
if uploaded_ref is not None:
    st.session_state.ref_upload = uploaded_ref
if uploaded_mask is not None:
    st.session_state.mask_upload = uploaded_mask
if uploaded_uv is not None and len(uploaded_uv) > 0:
    st.session_state.uv_uploads = uploaded_uv

# Button: Load default example images from assets
if st.sidebar.button("Load Example Images (Default)"):
    try:
        st.session_state.ref_upload = load_example_file("assets/reference.png", "reference.png")
        st.session_state.mask_upload = load_example_file("assets/mask.png", "mask.png")
        st.session_state.uv_uploads = [load_example_file("assets/uv_0.004000s_358.png", "uv_0.004000s_358.png")]
        st.sidebar.success("Example images loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading example images: {e}")

# Display names of uploaded files
if st.session_state.ref_upload:
    st.sidebar.info(f"Reference: {st.session_state.ref_upload.name}")
if st.session_state.mask_upload:
    st.sidebar.info(f"Mask: {st.session_state.mask_upload.name}")
if st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0:
    uv_names = ", ".join([uv.name for uv in st.session_state.uv_uploads])
    st.sidebar.info(f"UV Images: {uv_names}")

# ------------------------------
# Sidebar: Processing Parameters
# ------------------------------
st.sidebar.header("Processing Parameters")
scale = st.sidebar.slider("Difference Scale Factor", 0.1, 10.0, 1.0, step=0.1)
epsilon = st.sidebar.number_input("Normalization Epsilon", value=1e-6, format="%.7f")
iterations = st.sidebar.slider("Smoothing Iterations", 1, 100, 30, step=1)
ksize = st.sidebar.slider("Kernel Size (Odd)", 3, 101, 41, step=2)

params = {
    'scale': scale,
    'epsilon': epsilon,
    'iterations': iterations,
    'ksize': ksize
}

# ------------------------------
# Quick Pipeline Preview: Show uploaded images with stats
# ------------------------------
st.header("Quick Pipeline Preview")
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
    st.subheader("Mask")
    if st.session_state.mask_upload:
        st.image(Image.open(st.session_state.mask_upload), use_container_width=True)
        with st.expander("View Details"):
            st.markdown(format_file_stats(st.session_state.mask_upload))
    else:
        st.write("No Mask image loaded.")
with preview_cols[2]:
    st.subheader("UV")
    if st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0:
        # Show first UV image preview
        st.image(Image.open(st.session_state.uv_uploads[0]), use_container_width=True)
        with st.expander("View Details"):
            st.markdown(format_file_stats(st.session_state.uv_uploads[0]))
    else:
        st.write("No UV image loaded.")

# ------------------------------
# Main Processing Button and Display
# ------------------------------
if st.sidebar.button("Process Images"):
    if not st.session_state.ref_upload or not (st.session_state.uv_uploads and len(st.session_state.uv_uploads) > 0):
        st.error("Please upload at least a Reference Image and one UV Image (or load example images).")
    else:
        with st.spinner("Processing images aggressively..."):
            output = process_pipeline(st.session_state.ref_upload, st.session_state.mask_upload, st.session_state.uv_uploads, params)
        if output:
            st.success("Processing complete!")
            
            # ------------------------------
            # Display Intermediate Steps
            # ------------------------------
            st.header("Intermediate Steps")
            col_ref, col_mask = st.columns(2)
            with col_ref:
                st.subheader("Reference")
                st.image(safe_display(output['Reference']), caption="Normalized Reference", use_container_width=True)
            with col_mask:
                st.subheader("Mask")
                st.image(safe_display(normalize(output['Mask'])), caption="Mask (or Default)", use_container_width=True)
            
            st.subheader("Bounding Box")
            st.write(f"Coordinates: {output['BoundingBox']}")
            
            st.subheader("Cropped Reference")
            st.image(safe_display(output['CroppedReference']), caption="Cropped Reference", use_container_width=True)
            
            st.subheader("UV Details & Differences")
            for uv in output['UV_Details']:
                st.markdown(f"**{uv['name']}**")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(safe_display(uv['CroppedUV']), caption="Cropped UV", use_container_width=True)
                with c2:
                    st.image(safe_display(uv['Difference']), caption="Difference Map", use_container_width=True)
                    st.caption(f"Extracted Exposure: {uv['Exposure']}")
            
            st.subheader("Combined Difference")
            st.image(safe_display(output['CombinedDifference']), caption="Combined Difference", use_container_width=True)
            
            st.subheader("Refined & Inverse Masks")
            c1, c2 = st.columns(2)
            with c1:
                st.image(safe_display(output['RefinedMask']), caption="Refined Mask", use_container_width=True)
            with c2:
                st.image(safe_display(output['InverseMask']), caption="Inverse Mask", use_container_width=True)
            
            # ------------------------------
            # Final Outputs
            # ------------------------------
            st.header("🚀 Final Mask Outputs")
            final_cropped = output['FinalCroppedMask']
            final_full = output['FinalFullMask']
            col_final1, col_final2 = st.columns(2)
            with col_final1:
                st.image(safe_display(normalize(final_cropped.astype(np.float32)/255.0)), caption="🟢 Final Cropped Mask", use_container_width=True)
                st.download_button(
                    label="📥 Download Cropped Mask",
                    data=image_to_bytes(final_cropped),
                    file_name="final_cropped_mask.png",
                    mime="image/png"
                )
            with col_final2:
                st.image(safe_display(normalize(final_full.astype(np.float32)/255.0)), caption="🔵 Final Full Mask", use_container_width=True)
                st.download_button(
                    label="📥 Download Full Mask",
                    data=image_to_bytes(final_full),
                    file_name="final_full_mask.png",
                    mime="image/png"
                )
            
            st.success("🔥 Your customized UV mask is READY! Print like a pro. 🖨️✨")
