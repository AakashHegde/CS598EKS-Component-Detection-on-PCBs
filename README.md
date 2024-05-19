# Real-Time Component Identification on Printed Circuit Boards

### Read the full report [here](https://github.com/AakashHegde/CS598EKS-Component-Detection-on-PCBs/blob/main/RealTimeComponentIDOnPCBs.pdf).
### View the project poster [here](https://github.com/AakashHegde/CS598EKS-Component-Detection-on-PCBs/blob/main/RealTimeComponentIDOnPCBsPoster.pdf).

## Overview

This project focuses on addressing the issue of growing electronic waste by developing a real-time system for the detection and classification of electronic components on waste printed circuit boards (WPCBs). By utilizing deep learning models, specifically YOLO (You Only Look Once), we aim to facilitate the extraction and reuse of electronic components from WPCBs.

## Motivation

With the rapid growth in electronic device usage, electronic waste (e-waste) is increasing significantly. A large portion of this e-waste consists of WPCBs, which contain high-value and potentially reusable electronic components (ECs). However, the reuse of these components is limited due to the difficulty in identifying them efficiently. Our project seeks to improve this process by providing a cost-effective, real-time solution for component identification.

## System Components

The system is composed of three primary modules:

1. **Image Capture Module**: Captures real-time video feed from a camera overlooking the conveyor belt carrying WPCBs.
2. **Image Preprocessing Module**: Enhances the captured frames by segmenting them to improve the detection of small components.
3. **Object Detection and Classification Module**: Uses a deep learning-based model to detect and classify components within the preprocessed images.

## Methodology

- **Image Capture**: Utilizes an ordinary camera to capture high-resolution images at a frame rate of at least 30 frames per second.
- **Image Preprocessing**: Applies Slicing Aided Hyper Inference (SAHI) to segment images, enhancing the detection of small components.
- **Object Detection**: Employs YOLO models, integrated with SAHI, to detect and classify electronic components in real-time.

## Evaluation

The performance of the system is evaluated using mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds. The integration of SAHI with YOLO models has shown improved detection capabilities, particularly for small objects like electronic components on PCBs.

## Results

- The YOLOv8 model with SAHI achieved a mAP50 of 56.2 and a mAP50-95 of 38.4, indicating effective detection and classification of components.
- The system operates in real-time, making it suitable for practical applications in recycling facilities.

## Conclusion

Our system provides a feasible solution for the real-time identification of reusable electronic components on WPCBs. By making the recycling process more efficient and cost-effective, this project contributes to reducing electronic waste and promoting the reuse of valuable components.

## Dataset used
https://app.roboflow.com/firstworkspace-bv4k6/kuo-pcb-component-id/2