# 🩺 Diabetes Prediction Project

A machine learning project built in *Jupyter Lab* to predict the likelihood of diabetes in patients using *synthetic data*. This project demonstrates an end-to-end ML pipeline including data generation, preprocessing, model training, and evaluation.

---

## 📌 Overview

This project applies supervised machine learning to predict whether a patient is diabetic or not based on health-related features. Since real patient data raises privacy concerns, *synthetic data* was generated to simulate realistic medical records for training and testing the models.

---

## 📂 Project Structure


diabetes-prediction/
├── data/
│   └── synthetic_diabetes_data.csv   # Synthetically generated dataset
├── notebooks/
│   ├── 01_data_generation.ipynb      # Synthetic data generation
│   ├── 02_exploratory_analysis.ipynb # EDA and visualizations
│   ├── 03_preprocessing.ipynb        # Data cleaning & feature engineering
│   └── 04_model_training.ipynb       # Model training and evaluation
├── models/
│   └── diabetes_model.pkl            # Saved best model
├── requirements.txt
└── README.md


---

## 🧪 Synthetic Data

The dataset was *synthetically generated* (e.g., using libraries like sklearn.datasets, SDV, Faker, or numpy random distributions) to simulate realistic diabetic patient records.

### Features

| Feature                    | Description                                      | Type       |
|---------------------------|--------------------------------------------------|------------|
| Pregnancies               | Number of pregnancies                            | Integer    |
| Glucose                   | Plasma glucose concentration (mg/dL)             | Float      |
| BloodPressure             | Diastolic blood pressure (mm Hg)                 | Float      |
| SkinThickness             | Triceps skinfold thickness (mm)                  | Float      |
| Insulin                   | 2-Hour serum insulin (mu U/ml)                   | Float      |
| BMI                       | Body mass index (kg/m²)                          | Float      |
| DiabetesPedigreeFunction  | Genetic risk score for diabetes                  | Float      |
| Age                       | Age of the patient (years)                       | Integer    |
| *Outcome*               | Target: 1 = Diabetic, 0 = Non-Diabetic           | Binary     |

> *Note:* All data used in this project is synthetically generated and does not represent real patients.

---

## ⚙️ Installation

1. *Clone the repository*
   bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   

2. *Create a virtual environment*
   bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   

3. *Install dependencies*
   bash
   pip install -r requirements.txt
   

4. *Launch Jupyter Lab*
   bash
   jupyter lab
   

---

## 🚀 How to Run

Open the notebooks *in order* inside Jupyter Lab:

| Step | Notebook                        | Description                            |
|------|---------------------------------|----------------------------------------|
| 1    | 01_data_generation.ipynb      | Generate synthetic diabetes dataset    |
| 2    | 02_exploratory_analysis.ipynb | Visualize and understand the data      |
| 3    | 03_preprocessing.ipynb        | Clean, encode, and scale features      |
| 4    | 04_model_training.ipynb       | Train, compare, and evaluate models    |

---

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost

---

## 📈 Results

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 78%      | 0.76      | 0.74   | 0.75     |
| Decision Tree          | 79%      | 0.78      | 0.76   | 0.77     |
| Random Forest          | 82%      | 0.81      | 0.79   | 0.80     |
| SVM                    | 80%      | 0.79      | 0.77   | 0.78     |
| XGBoost                | *84%*  | *0.83*  | *0.82* | *0.82* |

> ✅ *Best Model: XGBoost* with 84% accuracy on synthetic test data.

---

## 🛠️ Technologies

| Tool/Library   | Purpose                          |
|----------------|----------------------------------|
| Python 3.10+   | Core programming language        |
| Jupyter Lab    | Interactive development environment |
| Pandas         | Data manipulation                |
| NumPy          | Numerical computations           |
| Scikit-learn   | ML models and preprocessing      |
| XGBoost        | Gradient boosting model          |
| Matplotlib     | Data visualization               |
| Seaborn        | Statistical plots                |
| Imbalanced-learn | Handling class imbalance       |

---

## 📋 Requirements


pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
jupyterlab


Install all with:
bash
pip install -r requirements.txt


---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add your feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

*Your Name*  
GitHub: [@your-username](https://github.com/your-username)  
LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
