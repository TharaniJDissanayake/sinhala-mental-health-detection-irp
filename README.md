Project Overview
- This project aims to detect mental health conditions, specifically depression, anxiety, and normal states, from Sinhala social media text using various machine learning and deep learning approaches.
The research focuses on low-resource language processing and explainable AI (XAI), addressing gaps in existing Sinhala mental health detection studies.

Six models were implemented and evaluated:

- Logistic Regression (LR)

- Support Vector Machine (SVM)

- Random Forest (RF)

- XGBoost (XGB)

- Convolutional Neural Network (CNN)

- XLM-RoBERTa (XLM-R) Transformer Model

= Explainability techniques (LIME, Feature Importance) were integrated to make the models interpretable.

Data

- The dataset was created by translating English mental health text datasets into Sinhala using secure and ethical translation practices.

- It includes Unicode Sinhala text, annotated into three classes: Depression, Anxiety, and Normal.

- Publicly available English datasets were ethically adapted, ensuring no private user information was involved.

Model Training

- Traditional Machine Learning models (LR, SVM, RF, XGB) were trained using TF-IDF features.

- The CNN model was trained with learned dense word embeddings.

- XLM-RoBERTa used a pre-trained transformer model fine-tuned on the Sinhala dataset.

Evaluation Metrics

- Accuracy

- Precision

- Recall

- F1-Score

- ROC-AUC

- Confusion Matrices

- Learning Curves

Explainability was ensured by:

- LIME for traditional models

- Token-level permutation importance for CNN

Tools and Libraries

- Python (v3.10)

- Scikit-learn

- TensorFlow / Keras

- Hugging Face Transformers

- Matplotlib, Seaborn for visualisation

- Imbalanced-learn (SMOTE for class balancing)

- Jupyter Notebook

- Google Colab and Mac M2 (local development)

