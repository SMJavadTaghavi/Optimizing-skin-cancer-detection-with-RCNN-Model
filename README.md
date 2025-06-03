

![image](https://github.com/user-attachments/assets/8ba6366a-1299-4885-80a6-2ad5108ffd47)



# 🧬 Optimizing Skin Cancer Detection with RCNN Model

### A Deep Learning Approach for Early and Accurate Skin Lesion Classification

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Skin_Cancer_Classification.jpg/800px-Skin_Cancer_Classification.jpg" width="70%" />
</div>

## 🔍 Introduction

Skin cancer remains a global health concern, with melanoma and squamous cell carcinoma posing life-threatening risks if not detected early. This repository presents an AI-powered diagnostic tool using advanced **Convolutional Neural Networks (CNN)** and **Recurrent CNNs (RCNN)** that aims to **optimize skin cancer detection through precise medical image classification**.

By leveraging **pre-trained deep learning models** such as ResNet50, EfficientNetB7, and hybrid CNN-LSTM architectures, this project enhances the accuracy, robustness, and interpretability of skin lesion classification tasks.

---

## 🎯 Objectives

* 📌 Develop and compare multiple CNN architectures (ResNet, VGG, EfficientNet) for skin lesion classification.
* 🔬 Integrate LSTM layers with CNNs to capture spatial and temporal dependencies.
* 📈 Apply Transfer Learning and Data Augmentation to improve generalization and prevent overfitting.
* 💡 Utilize explainability tools like **Grad-CAM** and **SHAP** for clinical transparency and decision support.

---

## 🧪 Dataset

We use a publicly available **Skin Cancer dataset from Kaggle**, containing thousands of **dermoscopic and clinical skin images** labeled as **benign or malignant**. The dataset includes:

* 📷 High-resolution medical images
* 🏷️ Annotated classes for supervised training
* 🔁 Balanced preprocessing for training fairness

---

## 🧠 Models Trained

| Model                 | Train Accuracy | Val Accuracy | Train Loss | Val Loss   |
| --------------------- | -------------- | ------------ | ---------- | ---------- |
| CNN (Basic)           | 95.44%         | 78.43%       | 0.1389     | 0.5256     |
| ResNet50              | 98.23%         | 88.73%       | 0.0470     | 0.3211     |
| VGG16                 | 66.78%         | 72.55%       | 0.5984     | 0.5430     |
| EfficientNetB7        | 100.00%        | 86.27%       | 0.0165     | 0.3148     |
| EfficientNetB7 + LSTM | 96.16%         | **94.12%**   | 0.1097     | **0.1626** |


![image](https://github.com/user-attachments/assets/95b082ce-2faa-49a1-ba61-991fa5878567)


![image](https://github.com/user-attachments/assets/58fbbb81-8a13-44c3-a5da-fe46ba9b72f3)

---

## 🛠 Technologies

* **Python 3.8+**
* **TensorFlow / Keras**
* **OpenCV & NumPy**
* **Matplotlib & Seaborn**
* **Grad-CAM & SHAP for Interpretability**

---

## ⚙️ How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/Optimizing-skin-cancer-detection-with-RCNN-Model.git
cd Optimizing-skin-cancer-detection-with-RCNN-Model

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Generate Grad-CAM outputs
python explain_model.py
```

---

## 📌 Conclusion

This project represents a step forward in **AI-based early cancer detection**, empowering healthcare systems with **automated, accurate, and interpretable** diagnostics. Our results demonstrate the effectiveness of combining powerful CNN backbones with sequence modeling using LSTM to capture complex image patterns in skin cancer diagnosis.

---

## 🧑‍⚕️ Citation

> If you use this work in your research, please cite the original Kaggle dataset and foundational paper:
> **"Prostate Cancer Detection Using Machine Learning" – Alzboon & Al-Batah (2023)**

![image](https://github.com/user-attachments/assets/f58ec088-f7f9-4b7e-9710-7e2acbc51321)

![image](https://github.com/user-attachments/assets/00290770-8cb3-4cf4-81c8-3ebe9377beec)

![image](https://github.com/user-attachments/assets/d0c849ec-eaea-4fa0-bccc-748a8bcda603)


---

## 🤝 Contact

Developed with 💙 by \[Javad Taghavi]
Feel free to contribute, raise issues, or contact us for collaborations.




