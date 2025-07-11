# 🧬 Genetic Disorder Prediction

**Live App**: [https://geneticdisorderprediction-shreyamahajan.streamlit.app](https://geneticdisorderprediction-shreyamahajan.streamlit.app)

This Streamlit application predicts genetic disorder subclasses using machine learning based on user-selected features from a genomic dataset.

> ⚠️ **Note:** For the predictor to function correctly, the **'Disorder Subclass'** input feature should be selected **last**.

---

## 📌 Project Overview

It uses a K-Nearest Neighbors (KNN) classifier trained on labeled data.

---

## 🗂 Dataset

- **Source**: Kaggle  
- **Link**: [Predict the Genetic Disorders Dataset](https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes/data)

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: scikit-learn, pandas, numpy
- **Model**: K-Nearest Neighbors (KNN)

---

## 🚀 Features

- Interactive multiselect for user-defined features
- Real-time prediction of genetic disorder subclass
- Sidebar disclaimer and usage tips
- Fully hosted and publicly accessible demo

---

## 📎 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/genetic_disorder_prediction.git
   cd genetic_disorder_prediction

2. Install dependencies:
    ```bash
   pip install -r requirements.txt

3. Place preprocessed_data.csv in the root directory.

4. Run the app:
 ```bash
   streamlit run app.py

 
