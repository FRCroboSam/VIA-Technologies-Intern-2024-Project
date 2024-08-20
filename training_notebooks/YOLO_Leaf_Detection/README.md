# YOLO Leaf Detect Notebook Info 

This notebook is for training a YOLOv8 Image Detection model 
that can detect leaves in images and return Bounding Box coordinates for them. 

## Usage Instructions 
Make sure this notebook and the zip file to the YOLO dataset is stored in the same folder, as it is in this repo. 

## Model Export Formats. 
Formats: .pt, float32.tflite, saved_model, int8.tflite, .mdla

### .pt and float32.tflite. 
These are the default formats provided by YOLO after you finish training the model called 'best.pt' and 'float32.tflite', located in runs/train{train_num}/weights/best_weights of your root directory (YOLO tells you which train_num after training finishes).

You can directly test the .pt model with YOLO('best.pt').
The float32.tflite model can be converted into .mdla format to test on the VIA Transforma Board. 

### Saved Model and Quantized int8.tflite. 
Quantizing the float32.tflite model int an int8 tflite reduces the size of the model, allowing for faster inference speeds on the VIA transforma board
when converted to .mdla. To convert the model to int8 tflite, first convert the float32.tflite into Keras' Saved Model format. 

Run yolov8_intquantization.py, located in demo_scripts/model_conversion of this notebook to convert a float32.tflite into Saved_Model then perform quantization to int8.tflite. 

### MDLA
MDLA is Mediatek's format for machine learning models to run on the VIA Transforma Board. 

Run the MDLA conversion command to convert from .tflite into .mdla so the model can be run on the VIA Transforma board.

ncc-tflite —arch mdla3.0 —opt-accuracy —relax-fp32 -O 3 model.tflite -o model.mdla
