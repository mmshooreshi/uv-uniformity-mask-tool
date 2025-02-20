# uv-uniformity-mask-tool ✨
![CoolParrot](https://cultofthepartyparrot.com/parrots/hd/parrot.gif)
![CoolGif](https://media.giphy.com/media/sltXBTQh2ogIFYwNNk/giphy.gif)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/-streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/-opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PoC](https://img.shields.io/badge/PoC-Proof--of--Concept-3EE9A1?style=for-the-badge)](./LICENSE)

> **Note:** This project is a Proof-of-Concept – built to showcase a complete, interactive workflow for generating customized mask images. Use it as a starting point and expand it for your resin 3D printing needs. (´・ω・`)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Why This Tool?](#why-this-tool)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contribution](#contribution)
- [Contact](#contact)
- [License](#license)

---

## Overview
The **uv-mask-tool** is a single-page web application that brings a full-detail UV mask generation pipeline directly into your browser. With an intuitive drag-and-drop interface, you can upload your reference image, mask image, and multiple UV images, customize every processing parameter, and visualize each step in real time. The end goal? To produce perfectly tweaked mask images for UV displays—ideal for ensuring precision when working with resin 3D printers.

---

## Features
- **Drag-and-Drop Interface:** Easily upload your reference, mask, and UV images.
- **Customizable Parameters:** Tweak exposure settings, difference scale, smoothing parameters, cropping, and more.
- **Step-by-Step Visualization:** Examine every processing stage—from raw input to the final mask—in expandable sections.
- **Downloadable Outputs:** Export your final, fully tweaked mask images (both cropped and full-size) ready for UV exposure applications.
- **Modern & Responsive:** Built with Streamlit for a sleek, user-friendly experience on any device.

---

## Why This Tool?
For resin 3D printing enthusiasts and professionals alike, achieving the perfect UV mask is critical for high-quality prints. The **uv-mask-tool** offers:
- **Precision:** Fine-tune every parameter to generate mask images tailored for optimal UV exposure.
- **Simplicity:** A one-page solution that wraps a complex image processing pipeline into an intuitive interface.
- **Efficiency:** Streamlined workflow with real-time previews, reducing trial-and-error in mask preparation.

---

## Getting Started
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/uv-mask-tool.git
   cd uv-mask-tool
   ```

2. **Set Up Your Environment:**
   - Ensure you have Python 3.7+ installed.
   - (Optional) Create a virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

5. **Open Your Browser:**
   The app will open automatically in your default web browser. If not, visit [http://localhost:8501](http://localhost:8501).

---

## Usage
- **Upload Files:** Use the sidebar to drag-and-drop your reference image, mask image, and one or more UV images.
- **Adjust Parameters:** Customize settings like the DIFF_SCALE_FACTOR, EPSILON, blur iterations, and kernel size to suit your resin printing requirements.
- **Process:** Click the “Process Images” button to generate your customized mask.
- **Explore Outputs:** Expand each section to view detailed intermediate steps and the final mask images.
- **Download Results:** Save your final masks with the provided download buttons—ready for UV display in resin 3D printing.

---

## Contribution
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.  
_Check out [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines._

---

## Contact
For any questions or suggestions, feel free to reach out on [Telegram @forthetim6being](https://t.me/forthetim6being).

---

## License
[![License](https://img.shields.io/github/license/yourusername/uv-mask-tool?style=for-the-badge)](./LICENSE)

---

Happy mask making and high-quality resin printing!  
Keep it lit and print on! ✨
