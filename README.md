# Houses Multimodal Price Prediction

## Objective
The objective of this project is to **predict house sale prices** using both structured/tabular data (bedrooms, bathrooms, area, location) and house images.  
The approach combines **CNN-based image feature extraction (MobileNetV2)** with tabular features in a **multimodal regression model**.  

---

## Methodology / Approach

### 1. Dataset
- Structured/tabular features: `bedrooms`, `bathrooms`, `area`, `zipcode`  
- Image features: 4 images per house from `Houses-dataset/Houses Dataset/`  
- Preprocessing:
  - Numeric: median imputation + standard scaling  
  - Categorical: most frequent imputation + one-hot encoding  

### 2. Tabular Preprocessing
- `ColumnTransformer` for combined numeric & categorical pipelines  
- Train, validation, test split  

### 3. Image Feature Extraction
- Pretrained **MobileNetV2** (without top layer)  
- Extract features for each house from 4 images  
- Average features across all available images  

### 4. Multimodal Model
- Tabular branch: Dense layers + Dropout  
- Image branch: Dense layers + Dropout  
- Combined branch: Dense layers + output regression node  
- Log-transform applied to target variable for stability  

### 5. Training & Evaluation
- Loss: MSE  
- Metrics: MAE, RMSE  
- Callbacks: EarlyStopping, ReduceLROnPlateau  
- Visualizations: Training/validation loss, actual vs predicted prices  

### 6. Export
- Model saved as `houses_multimodal_model.keras`  
- Tabular preprocessor saved as `houses_tabular_preprocessor.joblib`  

---

## Key Results / Observations
- Combining image features with tabular data improved prediction accuracy  
- MobileNetV2 transfer learning extracted rich visual representations  
- Multimodal approach achieved good **MAE** and **RMSE** compared to single-modality baselines  
- Preprocessing and modeling are modular and reusable for deployment  
- Model and preprocessor saved for production-ready use  

---

## Tech Stack
- Python 3.10+  
- TensorFlow/Keras  
- Scikit-learn  
- Pandas, NumPy, Matplotlib, Seaborn  
- Joblib  


