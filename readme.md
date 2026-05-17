# Employee Attrition Prediction Project

##  Project Objective
The objective of this project is to develop a predictive model that determines whether an employee is likely to leave the company(Attrition) by analyzing employee data.

---

##  Algorithm Used
We use a **classification algorithm (Logistic Regression)** to classify employees based on the dataset.  
The model predicts whether an employee will leave the company or not (**Yes/No**).

---

##  Dataset
- Dataset: IBM HR Analytics Dataset  
- Total Records: 1,470 employees  
- Total Features: 31  
- Target Variable: **Attrition (Yes/No)**  

---

##  Project Structure

### 1. `data_describe.ipynb`
- Provides dataset description  
- Identifies hidden patterns  
- Shows dataset balance  

---

### 2. `data_processing.py`
- Converts **Attrition** into numerical form  
- Converts **OverTime** into numerical values
- Coverts **BusinessTravel** into numerical values 
- Converts **MaritalStatus** into numerical values 
- Removes irrelevant features  
- Selects meaningful features  

#### 🔻 Dropped Features
- Department  
- EducationField  
- EmployeeCount  
- EmployeeNumber  
- Gender  
- JobRole  
- Over18  
- StandardHours  

####  Input Features (Independent Variables)
- DailyRate  
- DistanceFromHome  
- Education  
- EnvironmentSatisfaction  
- HourlyRate  
- JobInvolvement  
- JobLevel  
- JobSatisfaction  
- MonthlyIncome  
- NumCompaniesWorked  
- OverTime  
- PercentSalaryHike  
- PerformanceRating  
- TotalWorkingYears  
- WorkLifeBalance  
- YearsAtCompany  
- YearsInCurrentRole  
- YearsSinceLastPromotion  
- Age  
- RelationshipSatisfaction  
- StockOptionLevel  
- YearsWithCurrManager  
- BusinessTravel
- MaritalStatus
- MonthlyRate
- TrainingTimesLastYear 
####  Target Variable
- Attrition (Yes/No)

---

### 3. `Dataset.xlsx`
- Contains IBM HR dataset  
- 1,470 employee records  
- 31 features  

---

### 4. `main.py`
Contains the main implementation:

#### 🔹 Steps:
- Import libraries (`pandas`, `sklearn`)  
- Load processed dataset  
- Separate features and target variable  

#### 🔹 Data Splitting
- 80% training data  
- 20% testing data  
- `test_size = 0.2`, `random_state = 1`  

#### 🔹 Feature Scaling
- Standardization applied to input features  

#### 🔹 Model Training
- Logistic Regression  
- Parameters:
  - `class_weight = "balanced"`
  - `max_iter = 1000`

#### 🔹 Prediction
- Predict probabilities  
- Select probability of class **"Yes"**  
- Threshold = **0.44** → classify employee as leaving  

#### 🔹 Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Recall  
- Precision  
- F1 Score  

---

### 5. `Visualization.ipynb`
- Data visualization  
- Helps identify trends and patterns  

---

### 6. `requirements.txt`
Contains all required libraries: