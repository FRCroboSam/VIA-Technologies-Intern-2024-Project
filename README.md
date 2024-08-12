# VIA-Technologies-Intern-2024-Project
2025 VIA Technologies Edge AI Intern code for performing plant leaf detection, segmentation, and disease detection 
for Debian AI Transforma boards. 

leaf_detect_segment.py contains code for testing the 3 models in sequence -> first performing 

## Visualizing Results with leaf_detect_segment.py
leaf_detect_segment.py contains code for testing the 3 models in sequence -> first performing yolo bounding box detection 
to detect leaves, then taking the bounded areas of the image to segment the area with leaves, and finally 
detecting diseases. 

It then outputs the results in a folder called leaf_detect_results 
