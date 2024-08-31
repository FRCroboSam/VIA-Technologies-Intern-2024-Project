# VIA-Technologies-Intern-2024-Project
This repository contains the code for deploying and setting up plant leaf Image Detection, Image Segmentation, and Disease Detection models deployed on
the Debian AI Transforma boards. 

### Video Demo

Model Image Detection, Image Segmentation, disease detection demo on live video. 

https://github.com/user-attachments/assets/bdcdc902-1b9a-4088-8076-686b1e552d6b

Green bounding boxes means healthy, red means disease. Corresponding colored masks are also applied onto leaves,
making it easier to detect which ones have disease. 

## YOLOv8 Leaf Detection and MobileNet Image Segmentation Pipleine
The YOLO model detects the leaves, returning the detected area into 
the image segmentation model. The segmentation model processes the image, leaving only areas with the
leave intact and making the rest of the image black, feeding it into the disease detection model which returns the plant's disease.  

For the visualization, we overlay the detected bounding boxes, image masks (red if diseased, green if healthy), and sometimes display the name of the disease. Below our results on different types of images. 

### Results on Images from the Plant Village Dataset
![image](https://github.com/user-attachments/assets/5b74cf1e-e90b-44eb-99e2-7cfd94df6110)
![image](https://github.com/user-attachments/assets/b0f5d9a2-f310-4e98-affa-f504fe9b47a2)
![image](https://github.com/user-attachments/assets/df5c3d22-b7ab-4d04-abfc-ddc925930b31)

### Results on Live Video
![image](https://github.com/user-attachments/assets/a1b4a973-5bad-475d-9304-5ce75fcd040a)
![image](https://github.com/user-attachments/assets/47a0a626-1e30-4a48-bb01-85e16774efe2)
![image](https://github.com/user-attachments/assets/3df0c0e5-7610-4607-b95c-873a7a0b9552)
![image](https://github.com/user-attachments/assets/9cbdf8c8-ef99-4bb3-93a3-c3d71aff9cd7)

### Images taken in the office 
![image](https://github.com/user-attachments/assets/58e2054a-f51c-439d-bd88-132bcc0dc1a4)
![image](https://github.com/user-attachments/assets/7489aa7a-e9cb-414d-9a38-d8b170b312d5)
![image](https://github.com/user-attachments/assets/3a69c471-63e4-4fa2-9039-defe73f4b31a)

![image](https://github.com/user-attachments/assets/8de149f6-286e-4594-90dd-637dbae0c06d)

