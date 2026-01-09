# 🩺 Pneumonia Detection from Chest X-Rays: A Comparative Study of Custom CNN, VGG16, and ResNet50

## 📌 Project Overview
Early and accurate detection of pneumonia from chest X-rays is critical for patient outcomes. This project implements a deep learning-based diagnostic support system. As a Data Scientist and Industrial Engineer, I focused on maximizing **Recall (Sensitivity)** to minimize "False Negatives"—ensuring that no patient with pneumonia is overlooked by the model.

The project evaluates three different approaches:
1. **Custom CNN:** A baseline architecture built from scratch.
2. **VGG16 (Transfer Learning & Fine-Tuning):** Leveraging pre-trained weights with surgical fine-tuning of the final blocks.
3. **ResNet50 (Transfer Learning & Fine-Tuning):** Utilizing residual connections for deep feature extraction.

## 📊 Dataset Analysis
* **Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
* **Data Imbalance:** The training set consisted of 1,341 Normal and 3,875 Pneumonia cases (approx. 1:3 ratio).
* **Handling Imbalance:** Applied **Class Weights** during training to penalize misclassifications of the minority "Normal" class.
* **Preprocessing:** Images were resized to 224x224x3 and normalized to the [0, 1] range.

## 🛠️ Methodology & Fine-Tuning
The most significant performance boost came from **Fine-Tuning** the VGG16 model.
* **Initial Phase:** Only the top classifier layers were trained while the VGG16 base remained frozen.
* **Fine-Tuning Phase:** Unfroze the final convolutional block (`block5`) and retrained with a very low learning rate ($1 \times 10^{-5}$) to adapt pre-trained features to specific medical patterns.
* **Data Augmentation:** Used rotation, shifts, and zooms to prevent overfitting and improve the model's ability to generalize.

## 📈 Performance Comparison (Final Test Results)
The **Fine-Tuned VGG16** model outperformed others, achieving a balance between high precision and near-perfect recall.

| Model Architecture | Test Accuracy | Pneumonia Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Custom Base CNN** | 76.19% | 0.97 | 0.89 |
| **ResNet50 (Fine-Tuned)** | 81.00% | 0.96 | 0.87 |
| **VGG16 (Fine-Tuned) ⭐** | **93.00%** | **0.97** | **0.94** |

## 🔍 Explainable AI (Grad-CAM Visualization)
To build clinical trust, I implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**. This allows us to see the "heatmaps" of where the model is looking to make its decision.

* **Successful Diagnosis:** The model correctly focuses on pulmonary opacities.
* **False Positive Analysis:** Identified that some misclassifications occurred due to the model focusing on mediastinal (heart) density instead of lung tissue.

## 🚀 Key Insights
* **Metric Choice:** In medical diagnosis, **Recall** is more vital than Accuracy. My model captures **97% of pneumonia cases**.
* **Model Depth:** While ResNet is a more modern architecture, **VGG16** proved to be more stable and effective for this specific dataset size and domain.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, OpenCV, Matplotlib, Scikit-learn.
* **Infrastructure:** Google Colab (T4 & L4 GPUs).


Model Link : 

https://drive.google.com/drive/folders/1L5yijfBXTOZtW9n-shRkSOX89meS8RBU?usp=sharing
---
Developed by **İsmail Cem ÖZÇELİK**.
