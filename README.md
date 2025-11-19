# Skin Disease Classification Using CNNs

This project applies deep learning techniques to classify different types of skin diseases using medical images from the DermNet dataset. It includes both a baseline CNN and a transfer learning model using MobileNetV2.

---

## Technologies Used

| Technology          | Purpose                                 |
| :------------------ | :--------------------------------------- |
| Python              | Core development language               |
| TensorFlow / Keras  | CNN models and training                 |
| MobileNetV2         | Transfer learning feature extractor     |
| Pandas, NumPy       | Data handling                           |
| Matplotlib, Seaborn | Visualization of results                |
| Scikit-Learn        | Evaluation metrics + confusion matrix   |

---

## Dataset Overview

- Source: DermNet / Kaggle  
- ~23 disease classes  
- ~3,500 total images  
- Images vary in lighting, skin tone, resolution, and visual patterns  
- Dataset contains imbalance (some classes < 100 images)

**Note:** The dataset is NOT included in this repository due to size limits.

### Common classes:
- Acne and Rosacea  
- Eczema  
- Psoriasis  
- Nail Fungus  
- Herpes  
- Urticaria  
- Hair Loss  
- And many others

---

## Project Structure

(Indented so the markdown block stays valid)

    dermnet_skin_disease_classification
    ├── Notebooks
    │   ├── 01_data_overview.ipynb
    │   ├── 02_eda.ipynb
    │   ├── 03_cnn_baseline.ipynb
    │   ├── 04_pretrained_model.ipynb
    │   └── 05_evaluation.ipynb
    ├── Models
    │   ├── final_mobilenet_model.h5
    │   ├── final_mobilenet_model.keras
    │   └── final_skin_disease_model.keras
    ├── Docs
    │   └── data_dictionary.txt
    ├── Data/Raw
    │   └── READ ME.txt
    ├── READ Me.txt
    └── Requirements.txt

---

## Features

- Full dataset exploration and visual analysis  
- Preprocessing pipeline: resizing, normalization, RGB conversion  
- Data augmentation to improve generalization  
- Baseline CNN built from scratch  
- Transfer learning using MobileNetV2 for improved performance  
- Training curves for accuracy and loss  
- Confusion matrix and detailed classification metrics  
- Saved pretrained models ready for reuse  

---

## How to Run Locally

### 1. Install dependencies

    pip install -r Requirements.txt

### 2. Run notebooks

Open the .ipynb files inside the Notebooks folder using Jupyter or VS Code.

### 3. Dataset

Download the DermNet dataset from Kaggle and place it in the appropriate directory following the train/test folder structure.

---

## Model Performance Summary

### Baseline CNN
- Accuracy: ~25–30%  
- Mild overfitting  
- Limited performance due to complexity of diseases and dataset imbalance  

### MobileNetV2 (Transfer Learning)
- More stable training  
- Better validation loss  
- Learns faster and handles visual features more efficiently  
- Significant improvement over baseline CNN  

---

## Observations

- Some diseases look visually similar (eczema vs psoriasis), increasing confusion  
- Class imbalance impacts macro metrics  
- Lighting, zoom, and angle variation make the dataset challenging  
- Transfer learning was essential for reaching acceptable accuracy  
- Grad-CAM or explainability methods can help interpret predictions  

---

## Future Work

- Improve dataset balance with oversampling or class weights  
- Fine-tune deeper layers of MobileNetV2  
- Add stronger augmentations (color jitter, CutMix, MixUp)  
- Incorporate Grad-CAM visual explanations  
- Build a Streamlit web demo for real-time classification  
- Explore larger pretrained architectures  

---

## References & Credits

- Dataset: DermNet / Kaggle  
- Libraries: TensorFlow, Keras, NumPy, Scikit-Learn  
- Visualization: Matplotlib, Seaborn  

