# VIA-Technologies-Intern-2024-Project
2025 VIA Technologies Edge AI Intern code for performing plant leaf detection, segmentation, and disease detection 
for Debian AI Transforma boards. 

leaf_detect_segment.py contains code for testing the 3 models in sequence -> first performing 

## Visualizing Results with leaf_detect_segment.py
leaf_detect_segment.py contains code for testing the 3 models in sequence -> first performing yolo bounding box detection 
to detect leaves, then taking the bounded areas of the image to segment the area with leaves, and finally 
detecting diseases. 

It then outputs the results in a folder called leaf_detect_results 


## Training the Models

Training Yolo Leaf Detection Bounding Box Models (Leaf Detection) or Image Segmentatino models 

###Leaf Detection (YOLO_Leaf_Detect.ipnyb)
To run this notebook, make sure the zip folder to the dataset is stored in the 
same folder as the nobebook. TO unzip the dataset, make sure to leave the call to unzip() uncommented.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/85632f9f-b2df-4c6b-abfa-c78364cf8e2f">

The notebook also requires you to create an output_dir to store all the images, labels, and data.yaml file for training. 
In the provided notebook, I call it "Leaf_detect_output" but feel free to call it anything as long as it is in the same folder as 
the notebook. Make sure to create the yaml file in this folder as it is used for training.
<img width="589" alt="image" src="https://github.com/user-attachments/assets/4fe82f8c-9608-462e-9f05-0f6a7c2f8684">

Before training, make sure that the data.yaml path is absolute, otherwise there is difficulty with finding the 
train_images or val_images paths. 
<img width="998" alt="image" src="https://github.com/user-attachments/assets/b69b7207-696d-4330-81f2-702ce3a6da55">

After training the notebook, the folder structure should look something like this:
<img width="911" alt="image" src="https://github.com/user-attachments/assets/e45a0ca6-d09a-4fff-a6f1-f3d7ea47dc2c">
