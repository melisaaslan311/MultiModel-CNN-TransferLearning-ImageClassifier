## 🐾 Animal Classification Project: CNN & Transfer Learning

This project implements a robust deep learning pipeline using **PyTorch** for the classification of images into three classes: **Cats, Dogs, and Birds**. It features a custom-built Convolutional Neural Network (CNN) trained from scratch and three state-of-the-art Transfer Learning models (EfficientNet-B0, ResNet-18, MobileNetV2) that are fine-tuned on the dataset.

The project emphasizes rigorous **data preprocessing, augmentation, and comprehensive model evaluation** using advanced metrics and visualizations.

---

### 🎯 Project Objectives & Methodology

The project follows a structured, multi-day development cycle to ensure model stability and accurate performance assessment:

| Day(s) | Main Focus | Key Outcomes |
| :---: | :--- | :--- |
| **1-2** | **Data Preparation** | Standardized image formats, systematic file renaming (`cat_1.jpg`, etc.), created the **Train/Val/Test** folder structure with a balanced $64\%/16\%/20\%$ split. |
| **3** | **Preprocessing & Augmentation** | Resized images (244,244), normalized pixels to $[-1, 1]$. Applied **Random Affine/Rotation, Gaussian Noise, and Gaussian Blur** to the training set. |
| **4** | **Custom CNN Training** | Designed and trained a custom CNN on **Grayscale** images for 10 epochs. |
| **5** | **Transfer Learning (Fine-Tuning)** | Fine-tuned **EfficientNet-B0, ResNet-18, and MobileNetV2** models (ImageNet pre-trained) on **RGB** images for 5 epochs using GPU acceleration. |
| **6-7** | **Advanced Evaluation** | Implemented a comprehensive evaluation class to generate $15+$ metric visualizations, including Confusion Matrices, Confidence Analysis, and performance summaries for all four models. |

---


### 🧠 Model Architectures and Training

#### A. Custom CNN 

This model was built from scratch and optimized for **Grayscale (1-channel)** input.
$$
\text{Input (1x244x244)} \to [\text{Conv-BN-ReLU-Pool}] \times 4 \to \text{Dropout} \to \text{Linear (64} \times 15 \times 15) \to \text{Output (3)}
$$

* **Key Layers:** 4 convolutional blocks with **Batch Normalization (BN)** for stability.
* **Feature Map Progression (Channels):** $1 \to 8 \to 16 \to 32 \to 64$.
* **Final Layer:** The $64 \times 15 \times 15$ feature map is flattened and connected to the 3-class output layer.
* **Regularization:** $0.3$ Dropout applied before the final linear layer.
* **Training Config:** 10 Epochs, Adam optimizer ($lr=0.001$), CrossEntropyLoss.
  

#### B. Transfer Learning Models

Pre-trained models were fine-tuned using **RGB (3-channel)** input data for 5 epochs.

| Model | Architecture Highlights | Fine-Tuning Layer | Training Config |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | Efficient scaling of depth, width, and resolution. | `classifier[1]` layer modified $(1280 \to 3)$. | 5 Epochs, Adam ($lr=0.01$). |
| **ResNet-18** | Uses residual blocks to mitigate vanishing gradients. | `fc` layer modified $(512 \to 3)$. | 5 Epochs, Adam ($lr=0.01$). |
| **MobileNetV2** | Lightweight, uses depthwise separable convolutions. | `classifier[1]` layer modified $(1280 \to 3)$. | 5 Epochs, Adam ($lr=0.01$). |


<img width="750" height="200" alt="image" src="https://github.com/user-attachments/assets/fbd65723-05b0-472c-b878-2501604764ba" />
**EfficientNet-B0**
<img width="750" height="200" alt="image" src="https://github.com/user-attachments/assets/c1675e7e-a085-40c1-8cba-b98267e8ae5c" />
**ResNet-18** 
<img width="750" height="200" alt="image" src="https://github.com/user-attachments/assets/4bd15da2-ceb5-4897-aa14-863a51d0c140" />
**MobileNetV2**


---

### 📊 Advanced Model Evaluation 

A dedicated evaluation framework was used to provide deep insights into model performance beyond simple accuracy.

#### Evaluation Metrics & Visualizations

The evaluation generates a suite of visuals and data for each model:

1.  **Enhanced Confusion Matrix:** Visualizes raw counts and normalized percentages to identify misclassification patterns.
2.  **Metrics Dashboard:** Bar charts comparing **Precision, Recall, and F1-Score** per class, alongside data distribution.
3.  **Accuracy vs. Error Rate:** Class-wise visualization of correct vs. incorrect predictions.
4.  **Prediction Confidence Analysis:**
    * **Confidence Distribution:** Histograms showing the model's confidence level for correct vs. wrong predictions.
    * **Confidence Threshold Curve:** Plots how overall accuracy changes as the minimum required confidence threshold increases.
5.  **Performance Summary:** Radial plots and bar charts summarizing overall and class-wise accuracy.
6.  **Classification Report:** Detailed text report showing $\text{P, R, F1-Score}$ for each class, micro/macro/weighted averages, and support.
   <img width="921" height="417" alt="image" src="https://github.com/user-attachments/assets/42f7ef4e-2898-4476-85b6-d00e36f3c2bb" />


#### Output Files

Results are saved into dedicated folders for easy access and comparison:

* **Visualizations (.png):** All generated charts are saved.
* **Detailed Test Results (.csv):** Contains per-sample results: `True_Label`, `Predicted_Label`, `Correct`, and probabilities (`Prob_cats`, `Prob_dogs`, etc.).
* **Summary Statistics (.csv):** Aggregated metrics including **Overall Accuracy** and **Average Confidence**.
