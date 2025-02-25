# Brain Tumor Categorical Classifier

This repository contains a deep learning model to classify brain MRI scans into four different categories:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

## Project Overview

![gif](https://dixog9cdtdsmc.cloudfront.net/projects/brain-disease-classifier-demo.gif)

This project uses Convolutional Neural Networks (CNNs) to analyze MRI scans of the brain and classify them according to the type of tumor present, if any. The model is built using TensorFlow and Keras, specifically leveraging a pre-trained EfficientNetB0 architecture with transfer learning.
Also includes object detection.

### Dataset

![image](https://github.com/user-attachments/assets/82e7a1a3-4760-4921-b3c0-0a7425956a0e)
The dataset used in this project is sourced from Kaggle's "Brain Tumor MRI Dataset" by Masoud Nickparvar. 
It contains MRI scans categorized into the four classes mentioned above.

## Features

![image](https://github.com/user-attachments/assets/810f2134-f866-4ef9-b749-61133074d511)

- **Image Preprocessing**: Includes grayscale conversion, thresholding, erosion/dilation, and contour detection to standardize the input.
- **Data Augmentation**: Implements techniques like rotation, flipping, and brightness adjustment to improve model generalization.
- **Transfer Learning**: Utilizes a pre-trained EfficientNetB0 model fine-tuned on brain MRI data.
- **Performance Metrics**: Evaluates the model using accuracy, precision, recall, and F1-score.
- **Confusion Matrix Visualization**: Provides visual representation of model performance.

## Model Performance

![image](https://github.com/user-attachments/assets/358aa640-e093-4f44-99f7-83d434c1fd80)

The model achieves:
- Validation accuracy: 62.7%
- Test accuracy: 69.0%
- F1-Score: 0.857

## Requirements

The main dependencies include:
- TensorFlow
- Keras
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- imutils
- PIL

## Project Structure

```
brain-tumour-categorical-classifier/
├── brain-tumour.ipynb         # Main Jupyter notebook with code
├── requirements.txt           # Dependencies
└── data/                      # Dataset directory
    ├── Training/              # Training data
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    ├── Testing/               # Testing data
    └── data_cropped/          # Preprocessed images
```

## Usage

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook brain-tumour.ipynb`

## Model Architecture

- Base model: EfficientNetB0 (pre-trained on ImageNet)
- Additional layers:
  - Dropout (0.5)
  - Flatten
  - Dense (128 neurons, ReLU activation)
  - Output layer (4 neurons, softmax activation)

## Future Improvements

- Explore other pre-trained models like ResNet or VGG
- Implement more sophisticated data augmentation techniques
- Experiment with ensemble methods
- Increase dataset size for better generalization
- Implement cross-validation for more robust evaluation

## Acknowledgments

- Dataset provided by Masoud Nickparvar on Kaggle

---

*Note: This project is for educational purposes only and is not intended for clinical use or diagnosis.*
