This script trains a neural network on the Iris dataset, converts it to ONNX format, and performs inference using ONNX Runtime.

1. **Training the Model**:
   - Loads the Iris dataset using `sklearn.datasets.load_iris`.
   - One-hot encodes the target variable using `OneHotEncoder`.
   - Splits data into training (80%) and test (20%) sets with `train_test_split`.
   - Builds a Keras Sequential model with two hidden layers (10 neurons, ReLU activation) and an output layer (softmax activation).
   - Compiles the model with the Adam optimizer and categorical crossentropy loss.
   - Trains the model for 50 epochs with a batch size of 5.
   - Evaluates and saves the model in TensorFlow format to `./output/iris_model_tf`.

2. **Converting to ONNX**:
   - Uses `tf2onnx` to convert the TensorFlow model to ONNX format (opset 13).
   - Specifies input shape as `[None, 4]` (variable batch size, 4 features).
   - Saves the ONNX model to `./output/iris_model.onnx`.

3. **Inference with ONNX Runtime**:
   - Loads the ONNX model using `onnxruntime.InferenceSession`.
   - Prepares test data and a sample input array (3 samples, 4 features) for inference.
   - Runs predictions and converts output probabilities to class labels using `np.argmax`.
   - Calculates inference accuracy by comparing predicted and true labels.
   - Outputs predicted labels for the sample input.

This code demonstrates training a neural network, converting it to a portable ONNX format, and performing inference for deployment.
