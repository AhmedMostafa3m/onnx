
## YOLOv8 Object Detection with ONNX Export and Inference

This script demonstrates how to load a pre-trained YOLOv8 model, export it to ONNX format for cross-platform compatibility, and perform object detection inference on an image using the Ultralytics YOLO library.

### Code Explanation

1. **Import the YOLO Class**:
   - The `ultralytics.YOLO` class is imported to provide access to YOLOv8 model functionalities, including model loading, export, and inference.

2. **Load the YOLOv8 Model**:
   - A pre-trained YOLOv8 model (e.g., `yolov8n.pt`, the nano version) is loaded from the `./output/` directory using `YOLO("./output/yolov8n.pt")`.
   - The `.pt` file is a PyTorch model containing the trained weights for object detection.

3. **Export to ONNX Format**:
   - The `model.export(format="onnx", half=True)` method converts the YOLOv8 model to ONNX format.
   - The `half=True` parameter enables half-precision (FP16) to optimize the model for faster inference on compatible hardware (e.g., GPUs).
   - The exported ONNX model is saved as `./output/yolov8n.onnx`.

4. **Load the ONNX Model**:
   - The exported ONNX model is loaded using `YOLO("./output/yolov8n.onnx")` for inference.
   - The Ultralytics library supports seamless inference with ONNX models, abstracting the underlying ONNX Runtime details.

5. **Run Inference**:
   - The `onnx_model` performs object detection on an image retrieved from a URL (`https://ultralytics.com/images/bus.jpg`).
   - The `save=True` parameter saves the output (e.g., image with bounding boxes) to the default output directory.
   - The `show=True` parameter (optional) displays the output image with detected objects during execution.
   - The `results` object contains detection details, such as bounding boxes, class labels, and confidence scores, which are printed.

### Usage

To use this script:
1. Install the required dependency: `pip install ultralytics`.
2. Ensure the pre-trained YOLOv8 model (`yolov8n.pt`) is available in the `./output/` directory. You can download it from the Ultralytics YOLOv8 repository or train your own model.
3. Run the script to:
   - Export the YOLOv8 model to ONNX format (`./output/yolov8n.onnx`).
   - Perform object detection on the specified image.
   - Save and optionally display the results with detected objects.
4. The ONNX model can be deployed in any ONNX-compatible environment for efficient inference.

### Example Output
- The script generates an ONNX model file (`yolov8n.onnx`).
- Inference results include bounding boxes, class labels, and confidence scores for detected objects in the input image.
- The processed image with annotations is saved to the default output directory (e.g., `runs/detect/`).

### Notes
- The `yolov8n.pt` model is the nano version of YOLOv8, optimized for speed and efficiency. Other variants (e.g., `yolov8s.pt`, `yolov8m.pt`) can be used for different performance-accuracy trade-offs.
- The `half=True` export option requires compatible hardware (e.g., CUDA-enabled GPUs) for FP16 inference.
- Ensure an internet connection to download the sample image from the provided URL.

This script provides a straightforward pipeline for exporting a YOLOv8 model to ONNX and performing object detection, leveraging the portability of ONNX for deployment across various platforms.

