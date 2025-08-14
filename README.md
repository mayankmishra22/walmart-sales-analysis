# Walmart Sales Analysis –

### Project Overview
This project predicts the **weekly sales** of Walmart stores using historical data and external factors.  
The aim is to build a robust forecasting system to support:
- **Inventory management**
- **Staffing decisions**
- **Promotional planning**

A trained model is deployed locally with **Flask**, providing a simple web form for sales prediction.

---

## Data
Three datasets are used and merged into a comprehensive dataset for analysis and modeling:

1. **train.csv** – Historical sales for 45 stores  
   - Store number, department number, date, weekly sales, holiday flag.
2. **features.csv** – Store/date-level external features  
   - Temperature, fuel price, CPI, unemployment rate, markdown events.
3. **stores.csv** – Store-level static attributes  
   - Store number, type, and size.

After merging, new features are engineered:
- Month, year, day of the week
- Weekend indicator
- One-hot encoding for store type  
Negative weekly sales entries are removed.

---

## Methodology

1. **Data Loading & Exploration**  
   - Understanding dataset structure, handling missing values, and visualizing distributions.
   
2. **Data Preprocessing**  
   - Missing value handling for markdowns  
   - Date conversion to datetime format  
   - Boolean encoding for holidays  
   - One-hot encoding for store type

3. **Data Merging**  
   - Merging train.csv, features.csv, and stores.csv on common keys.

4. **Model Selection & Training**  
   - **Random Forest Regressor** – Final deployed model.  
   - **XGBoost Regressor** – Gradient boosting approach for regression.  
   - **ARIMA** – Time series statistical model via auto_arima.

5. **Model Evaluation**  
   - Metrics: MSE, RMSE, MAE  
   - Cross-validation for Random Forest  
   - Performance comparison table for all models

6. **Model Saving & Deployment**  
   - Best-performing Random Forest model saved as rf_model.pkl  
   - Flask application for interactive predictions

---

## Machine Learning Models Used
- **Random Forest Regressor** – Handles non-linear relationships and complex feature interactions.
- **XGBoost Regressor** – Optimized gradient boosting algorithm for speed and accuracy.
- **ARIMA** – Suitable for capturing time-based trends in sales.

---

## Flask App Overview
- app.py loads:
  - **rf_model.pkl** – Trained model
  - **merged_data.csv** – Preprocessed dataset
- User inputs:
  - Store ID  
  - Department number  
  - Store size  
  - Temperature  
  - Holiday flag  
  - Date
- Returns the predicted weekly sales for the selected week.

---

## Libraries Used
- **numpy**, **pandas** – Data manipulation  
- **matplotlib**, **seaborn** – Visualization  
- **scikit-learn** – ML models & preprocessing  
- **statsmodels**, **pmdarima** – Statistical modeling & ARIMA  
- **xgboost** – XGBoost model  
- **flask** – Web app deployment  
- **prettytable** – Tabular result display  

---

## Results
The models were compared based on **MSE**, **RMSE**, and **MAE**.  
Random Forest emerged as the most balanced performer and was selected for deployment.

---

## Folder Structure
```
.
├── app.py                       # Flask app for deployment
├── Walmart sales forecasting.ipynb  # Data processing & model training
├── rf_model.pkl                  # Trained Random Forest model
├── merged_data.csv               # Final processed dataset
├── templates/
│   └── index.html                # Web interface
├── static/
│   └── images
      └── walmarticon.png
│   └── css
      └── styles.css 
└── README.md
```

---

## How to Run Locally
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
**Example requirements.txt**
```
flask
pandas
numpy
scikit-learn
xgboost
statsmodels
pmdarima
matplotlib
seaborn
prettytable
```
### 3. Run Flask App
```bash
python app.py
```
Access at:
```
http://127.0.0.1:5000/

### Demo
<img width="1920" height="1080" alt="output" src="https://github.com/user-attachments/assets/4104db22-eed9-4ce3-924e-b0135e34eeeb" />


---

