# Face and Eye Detection Using Haarcascade Model in Google Colab

## Overview
This project demonstrates how to perform face and eye detection using the Haarcascade model in Google Colab. We'll use OpenCV's pre-trained Haarcascade classifiers to detect faces and eyes in images.

## Steps

1. **Setting up Google Colab**:
   - Head over to Google Colab and create a new notebook.
   - This platform allows you to run Python code and experiment with machine learning models.

2. **Importing Required Libraries**:
   - In a new Colab cell, import the necessary libraries:
   - Remember that it is optional to use matplotlib as per your liking (personally I didn't make use of it)
     ```python
     import cv2
     import matplotlib.pyplot as plt
     ```

3. **Loading the Haarcascade XML File**:
   - The Haarcascade algorithm requires a pre-trained XML file for face detection.
   - Download the Haarcascade XML file from the OpenCV GitHub repository:
     ```python
     !wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
     ```

4. **Loading and Displaying an Image**:
   - Upload an image to Google Colab using:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
   - Load and display the image:
     ```python
     img_path = list(uploaded.keys())[0]
     image = cv2.imread(img_path)
     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     plt.imshow(image_rgb)
     plt.axis('off')
     plt.show()
     ```

5. **Face Detection with Haarcascade**:
   - Initialize the Haarcascade classifier and detect faces in the image:
     ```python
     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
     ```

6. **Displaying Detected Faces**:
   - Highlight the detected faces by drawing rectangles around them:
     ```python
     for (x, y, w, h) in faces:
         cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
     plt.imshow(image_rgb)
     plt.axis('off')
     plt.show()
     ```

7. **Results**:
   - You have successfully implemented face detection using the Haarcascade algorithm in Google Colab.
   - The step-by-step process includes setting up Colab, importing libraries, loading the Haarcascade XML file, detecting faces, and displaying the results.

## References
- Building a Face Detection Model with Haar Cascade in Google Colab
- OpenCV Haarcascades Repository
