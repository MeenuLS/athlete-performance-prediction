# 🏃 Athlete Performance Prediction Using Machine Learning

A machine learning project that explores athlete performance data, trains regression models, evaluates predictive performance, and provides an interactive web application for exploring data and generating predictions.

## 📌 Project Overview

Athlete performance can be influenced by multiple factors, including training intensity, sleep, hydration, heart rate, body composition, and previous competition performance.

This project uses machine learning regression techniques to investigate relationships between athlete characteristics and a numerical performance metric. A Streamlit frontend provides an interactive interface for dataset exploration and performance prediction using a trained model.

## 🎯 Objectives

- Analyze athlete performance data using Python.
- Perform data preprocessing and exploratory data analysis (EDA).
- Train and evaluate machine learning regression models.
- Compare model performance using appropriate regression metrics.
- Save the trained model for reuse.
- Develop an interactive Streamlit application for predictions.

## 📊 Dataset

The dataset contains athlete-related information, training measurements, and a numerical target variable.

**Target variable:** `Performance_Metric`

Example features include:

- `Sport_Type`
- `Event`
- `Training_Hours_per_Week`
- `Average_Heart_Rate`
- `BMI`
- `Sleep_Hours_per_Night`
- `Daily_Caloric_Intake`
- `Hydration_Level`
- `Injury_History`
- `Previous_Competition_Performance`
- `Training_Intensity`
- `Resting_Heart_Rate`
- `Body_Fat_Percentage`
- `VO2_Max`
- `Event_Distance`
- `Altitude_Training`
- `Mental_Focus_Level`

The exact input features depend on the columns retained during model training.

**Dataset note:** Include the original dataset source and license in this section before publishing. If the dataset is hosted externally or contains data you are not permitted to redistribute, provide the source link rather than uploading the dataset itself.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Joblib / Pickle** — Model and metadata serialization
- **Streamlit** — Interactive web application
- **Jupyter Notebook / Google Colab** — Model development and experimentation
- **Git and GitHub** — Version control and project hosting

## 🤖 Machine Learning Workflow

1. Load and inspect the dataset.
2. Clean and preprocess the data.
3. Explore feature distributions and relationships.
4. Prepare the input features and target variable.
5. Train regression models.
6. Evaluate predictions on held-out data.
7. Save the selected trained model and required feature metadata.
8. Load the saved model in the Streamlit application.
9. Generate predictions through the interactive interface.

## 📏 Evaluation Metrics

The models can be evaluated using the following metrics:

| Metric | Description |
|---|---|
| R² Score | Measures how well the model explains variation in the target |
| MAE | Mean Absolute Error; average absolute prediction error |
| RMSE | Root Mean Squared Error; penalizes larger errors more heavily |

The actual evaluation results should be reported after running the final notebook. Model performance depends on the dataset, selected features, preprocessing, and model configuration.

## 🖥️ Streamlit Web Application

The frontend is designed to make the project easier to interact with without requiring users to run the training notebook.

Depending on the features enabled in the application, users can:

- Explore the uploaded athlete performance dataset.
- View dataset information and descriptive statistics.
- Train and evaluate regression models.
- Load a previously trained model.
- Enter feature values and generate performance predictions.

## 📁 Project Structure

```text
athlete-performance-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── athlete_performance_model.pkl
├── feature_names.pkl
├── model_name.pkl
│
├── notebooks/
│   └── athlete_performance_prediction.ipynb
│
└── data/
    └── sports_performance_data.csv
```

*This is a suggested structure. Adjust the filenames and folders to match the files actually included in your repository. Do not upload the dataset or model files unless you have confirmed that doing so is appropriate.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/MeenuLS/athlete-performance-prediction.git
cd athlete-performance-prediction
```

Replace the repository URL and folder name if your actual GitHub repository uses a different name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the main libraries:

```bash
python -m pip install streamlit pandas numpy matplotlib scikit-learn joblib
```

Use a Python version compatible with your installed dependencies and the saved model.

### 4. Prepare the required files

Place `app.py`, the saved model, and its required metadata in the locations expected by the application.

For the separate-file model-loading setup, the files are:

- `athlete_performance_model.pkl`
- `feature_names.pkl`
- `model_name.pkl`

Upload or select the dataset if the application requires it.

**Important:** The saved model must include or be used with the same preprocessing steps applied during training. The feature names and order must match those expected by the trained model.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 🔒 Data and Model Security

- Do not upload private datasets, credentials, API keys, or personal information.
- Only load trusted `.pkl` or `.joblib` files because loading these formats can execute malicious code.
- Check dataset redistribution permissions before publishing data.
- Ensure that the model's preprocessing and input schema are consistent between training and inference.

## ⚠️ Limitations

- Predictions are estimates and may not reflect actual athlete performance.
- Results depend on dataset quality, feature relevance, and model generalization.
- A strong evaluation score on training data does not guarantee accurate predictions on unseen data.
- This project is intended for educational and machine learning demonstration purposes, not as a substitute for professional sports science or medical assessment.

## 🚀 Future Improvements

- Improve feature engineering and data quality.
- Compare additional regression algorithms and tune hyperparameters.
- Add interactive visualizations and model comparison charts.
- Track model versions and evaluation results.
- Deploy the Streamlit application online.
- Investigate feature importance and model explainability.

## 👩‍💻 Author

**Meenu L S**

- GitHub: [@MeenuLS](https://github.com/MeenuLS)

## 📄 License

Add a license that matches your intended use and the permissions associated with the dataset and other project assets. Do not claim ownership of third-party data or code.
