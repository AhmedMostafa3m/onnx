This script demonstrates training a Random Forest Classifier on the Iris dataset, saving the model, converting it to ONNX format, and performing inference using ONNX Runtime.

1. **Training and Saving the Model**:
   - Loads the Iris dataset using `sklearn.datasets.load_iris`.
   - Splits the data into training and test sets with `train_test_split`.
   - Trains a Random Forest Classifier (`RandomForestClassifier`) with 100 estimators.
   - Saves the trained model to `output/model.pkl` using `joblib` with compression.

2. **Converting to ONNX**:
   - Loads the saved model using `joblib`.
   - Defines the input type as `FloatTensorType([None, 4])`, where `None` allows a variable batch size and `4` represents the four features of the Iris dataset.
   - Converts the model to ONNX format using `skl2onnx.convert_sklearn`.
   - Saves the ONNX model to `output/model.onnx`.

3. **Inference with ONNX Runtime**:
   - Loads the ONNX model using `onnxruntime.InferenceSession`.
   - Prepares sample input data as a NumPy array with shape `(3, 4)` (three samples, four features).
   - Runs inference by passing the input data to the model and retrieves predictions.
   - Outputs the predicted class labels.

This code enables model training, conversion to a portable ONNX format, and inference for deployment in various environments.

