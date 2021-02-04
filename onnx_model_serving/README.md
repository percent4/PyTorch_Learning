the repo shows how to serve models using ONNX Runtime Server.

### Author

- jclian91

### Code Structure

```
.
├── assets
├── captcha_recognition_model_predict_using_onnx_runtime_server.py
├── captcha_recognition.onnx
├── check_onnx_model.py
├── cls_model_predict_using_onnx_runtime_server.py
├── __init__.py
├── iris_model.py
├── iris.onnx
├── onnx_runtime_server.ipynb
├── onxx_model_predict.py
├── README.md
├── requirements.txt
├── ssd_model_predict_using_onnx_runtime_server.py
└── ssd.onnx

1 directory, 14 files

```

### ONNX Model Deployment

use Docker to deploy onnx model which PyTorch can export, bash command as follows:

```
docker run -v {localModelAbsoluteFolder}:{dockerModelAbsoluteFolder} -p {your_local_port}:8001 {imageName} --model_path {dockerModelAbsolutePath}
```

### ONNX HTTP service

- serving SSD Model

references: `onnx_runtime_server.ipynb`

- serving IRIS dataset MLP multiclass classification Model

see scripts: `onnx_model_predict.py`, `check_onnx_model.py`, `iris_model_predict_using_onnx_runtime_server.py`.

- serving captcha recognition CNN Model

see scripts: `captcha_recognize/torch_model_train.py`, `captcha_recognition_model_predict_using_onnx_runtime_server.py`

- serving transformers BERT text classification Model

see scripts: `transformers_learning/text_classification/model_export.py`, `cls_model_predict_using_onnx_runtime_server.py`


### References
1. https://github.com/onnx/tutorials
2. SSD model: https://onnxzoo.blob.core.windows.net/models/opset_10/ssd/ssd.onnx
3. SSD assets: https://github.com/onnx/tutorials/tree/master/tutorials/assets
4. Inferencing SSD ONNX model using ONNX Runtime Server: https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb
5. ONNX model online review: https://netron.app/
6. ONNX Protobuf files: https://github.com/microsoft/onnxruntime/blob/master/server/protobuf/onnx-ml.proto