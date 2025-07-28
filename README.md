# Brain Tumor Image Classification with CNN & Transfer Learning

This project focuses on classifying brain MRI images into four tumor types using both a **Custom Convolutional Neural Network (CNN)** and **Transfer Learning** models like **MobileNetV2**. A Streamlit app is also provided for interactive deployment.

## 🧠 Tumor Classes

* **Glioma**
* **Meningioma**
* **Pituitary**
* **No Tumor**

## 📁 Dataset

The dataset should be organized as follows:

```
brainTumor/
└── Tumour/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── no_tumor/
```
📌 Dataset
Source: [Brain Tumor MRI Multi-Class](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) 

```

## 🔧 Setup Instructions

1. **Clone this repository:**

```bash
https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Train the models (optional):** The notebook trains two models:

* A Custom CNN model
* A MobileNetV2 model with ImageNet weights

Trained weights will be saved in the `models/` directory.

4. **Run the Streamlit App:**

```bash
streamlit run app.py
```

## 🧪 Models Used

* ✅ **Custom CNN**: Built from scratch with convolution, pooling, batch normalization, dropout, and dense layers.
* ✅ **MobileNetV2**: Pre-trained on ImageNet and fine-tuned for brain tumor classification.

## 📈 Performance Metrics

Example output for MobileNetV2:

```
Accuracy: 0.92
Precision: 0.92
Recall: 0.92
F1-score: 0.92
```

## 📊 Streamlit Interface

* Upload an MRI image (JPEG, PNG)
* Choose model: Custom CNN or MobileNetV2
* Get predicted tumor type instantly

## 📂 Folder Structure

```
.
├── models/              # Saved model weights (.h5)
├── brainTumor/          # Dataset directory
├── app.py               # Streamlit application
├── brain.py       # Full training + evaluation 
├── README.md            # Project readme
└── requirements.txt     # Python dependencies
```

## 📌 Requirements

* TensorFlow
* Keras
* Streamlit
* NumPy, Pandas, Matplotlib, Seaborn
* scikit-learn


Feel free to open issues or submit pull requests if you'd like to improve or extend this work!
