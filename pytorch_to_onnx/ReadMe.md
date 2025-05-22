
## Iris Classification with PyTorch, ONNX Conversion, and Inference

This script trains a neural network using PyTorch on the Iris dataset, converts the trained model to ONNX format for portability, and performs inference using ONNX Runtime. The code is divided into three main sections: model training, conversion to ONNX, and inference.

### Section 1: Training a Simple Neural Network

This section loads the Iris dataset, prepares the data, defines a neural network, trains it, evaluates its performance, and saves the model in PyTorch format.

1. **Import Libraries**:
   - `torch`, `torch.nn`, `torch.optim`: PyTorch libraries for building and training neural networks.
   - `sklearn.datasets.load_iris`: Loads the Iris dataset.
   - `sklearn.model_selection.train_test_split`: Splits data into training and test sets.
   - `sklearn.preprocessing.OneHotEncoder`: Converts categorical labels to one-hot encoded format.
   - `torch.utils.data.DataLoader`, `TensorDataset`: Utilities for batching and handling data in PyTorch.

2. **Load and Prepare the Dataset**:
   - The Iris dataset is loaded using `load_iris()`, providing 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and 3 target classes (Iris species).
   - The target variable `y` is reshaped to `(-1, 1)` for compatibility with `OneHotEncoder`.
   - The target labels are one-hot encoded (e.g., class 0 becomes `[1, 0, 0]`) to suit multi-class classification.
   - The dataset is split into 80% training and 20% testing sets using `train_test_split` with a fixed `random_state=42` for reproducibility.

3. **Convert Data to PyTorch Tensors**:
   - Features (`X_train`, `X_test`) and labels (`y_train`, `y_test`) are converted to PyTorch tensors with `dtype=torch.float32` for compatibility with PyTorch's neural network operations.

4. **Create DataLoader**:
   - A `TensorDataset` is created from `X_train` and `y_train` to pair features with labels.
   - A `DataLoader` is used to batch the training data (batch size = 5) and shuffle it for better training dynamics.

5. **Define the Neural Network**:
   - A `SimpleNN` class is defined, inheriting from `nn.Module`.
   - The network has three fully connected layers:
     - `fc1`: Input layer (4 features) to 10 neurons with ReLU activation.
     - `fc2`: Hidden layer (10 neurons to 10 neurons) with ReLU activation.
     - `fc3`: Output layer (10 neurons to 3 neurons, one per class) with softmax activation for probability distribution.
   - The `forward` method defines the data flow through the layers.

6. **Set Up Loss and Optimizer**:
   - The model is initialized as `model = SimpleNN()`.
   - The loss function is `CrossEntropyLoss`, which combines log-softmax and negative log-likelihood loss, suitable for multi-class classification.
   - The optimizer is Adam with a learning rate of 0.001.

7. **Train the Model**:
   - The model is trained for 50 epochs.
   - For each epoch, the `DataLoader` provides batches of inputs and targets.
   - For each batch:
     - Gradients are reset (`optimizer.zero_grad()`).
     - The model computes predictions (`outputs`).
     - Loss is calculated between predictions and true labels (using `argmax` on one-hot encoded labels).
     - Gradients are computed (`loss.backward()`), and the optimizer updates the model parameters (`optimizer.step()`).
   - Loss is printed every 10 epochs for monitoring.

8. **Evaluate the Model**:
   - The model is set to evaluation mode using `torch.no_grad()` to disable gradient computation.
   - Predictions are made on the test set (`X_test`).
   - Predicted and true labels are compared using `torch.max` to extract class indices.
   - Accuracy is calculated as the fraction of correct predictions and printed.

9. **Save the Model**:
   - The model's state dictionary (weights and biases) is saved to `./output/iris_model_pytorch.pth` using `torch.save`.

---

### Section 2: Conversion to ONNX

This section converts the trained PyTorch model to ONNX format for cross-platform compatibility.

1. **Import Libraries**:
   - `torch.onnx`: Provides tools to export PyTorch models to ONNX format.

2. **Load the Saved Model**:
   - A new instance of `SimpleNN` is created and loaded with the saved state dictionary from `./output/iris_model_pytorch.pth`.
   - The model is set to evaluation mode (`model.eval()`) to ensure consistent behavior during export.

3. **Export to ONNX**:
   - A dummy input tensor (shape `[1, 4]`, matching one sample with 4 features) is created to trace the model's computation graph.
   - The `torch.onnx.export` function converts the model to ONNX format:
     - `input_names=['input']` and `output_names=['output']` label the input and output tensors.
     - `dynamic_axes` specifies that the batch size (dimension 0) is variable for both input and output, enabling flexible batch inference.
     - `opset_version=13` ensures compatibility with ONNX operators from version 13.
   - The ONNX model is saved to `./output/iris_model.onnx`.

4. **Confirmation**:
   - A message confirms the successful export of the ONNX model.

---

### Section 3: Inference with ONNX Runtime

This section performs inference using the ONNX model and ONNX Runtime.

1. **Import Libraries**:
   - `onnxruntime`: Provides tools for running inference with ONNX models.
   - `numpy`: Used for handling input data arrays.

2. **Load the ONNX Model**:
   - The ONNX model is loaded from `./output/iris_model.onnx` using `ort.InferenceSession`.

3. **Prepare Input Data**:
   - A sample input array with three samples (shape `[3, 4]`) is defined, representing three Iris samples with 4 features each.
   - The `prepare_input` function converts the input data to a dictionary with the input name (obtained from `onnx_model.get_inputs()[0].name`) and ensures the data type is `np.float32`.

4. **Perform Inference**:
   - The `onnx_model.run` method executes inference, passing the input data dictionary and requesting all outputs (`None`).
   - Predictions are extracted as a NumPy array containing probabilities for each class.

5. **Process Predictions**:
   - The `np.argmax` function converts the output probabilities to class labels by selecting the index of the highest probability for each sample.
   - The predicted labels are printed.

---

### Usage

To use this script:
1. Ensure dependencies (`torch`, `sklearn`, `onnxruntime`, `numpy`) are installed.
2. Run the script to:
   - Train a neural network on the Iris dataset.
   - Save the PyTorch model to `./output/iris_model_pytorch.pth`.
   - Convert it to ONNX format and save to `./output/iris_model.onnx`.
   - Perform inference on sample data and print predicted labels.
3. The ONNX model can be deployed in any ONNX-compatible environment for inference.

### Example Output
- Training loss is printed every 10 epochs (e.g., `Epoch [50/50], Loss: 0.1234`).
- Test accuracy (e.g., `Test Accuracy: 0.9667`).
- Predicted labels for the sample input (e.g., `Predicted labels: [1 0 2]`).

This script provides a complete pipeline for training, converting, and deploying a neural network for Iris classification, leveraging PyTorch for training and ONNX for portability.

