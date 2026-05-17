#streamlit Library is used for creating front end by using python.
import streamlit as st
#joblib is used for save and load machine learning and scalar(scalar : scale the input features).
import joblib
#Numpy library  is used for arrays.
import numpy as np

#Load Machine learning model in **model** object.
model = joblib.load("employee_attrition_model.pkl")
#Load Standard_Scalar in **scaler** object.
scaler = joblib.load("scaler.pkl")
 #                             Streamlit library Archetecture for this project

#  st.title() --> write title on web page in large and bold.
#  st.write() --> write a text on web page.
#  st.number_input(feature, min, max) --> input box of integer with range from (min to max)sssss and return integer.
#                                          Integer input box is used for large range.
#  st.slider(feature, min, max) --> slider with range (min to max) and return integer. Slider is used for small range.
#  st.selectbox(feature, [cat 1, Cat2]) --> create dropdown list. It return text further
#                                           we encode text into numeric format.
#  var = {cat1 : 1, cat2 : 2} --> all text (return by dropdown list) with their numerical values store in variable.
#  var[feature] --> Encode text (return by dropdown list) into respective numerical value by using variable **var**.
#  st.button("button name) --> when button is clicked, it return **True**.


st.title("Employee Attrition Prediction System")

st.write("Enter employee information below:")


age = st.number_input("Age", 18, 60)

business_travel = st.selectbox(
    "Business Travel",
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
)

daily_rate = st.number_input("Daily Rate", 102, 1499)

distance_from_home = st.number_input("Distance From Home", 1, 29)

education = st.slider("Education", 1, 5)

environment_satisfaction = st.slider("Environment Satisfaction", 1, 4)

hourly_rate = st.number_input("Hourly Rate", 30, 100)

job_involvement = st.slider("Job Involvement", 1, 4)

job_level = st.slider("Job Level", 1, 5)

job_satisfaction = st.slider("Job Satisfaction", 1, 4)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

monthly_income = st.number_input("Monthly Income", 1009, 19999)

monthly_rate = st.number_input("Monthly Rate", 2094, 26999)

num_companies_worked = st.number_input("Number of Companies Worked", 0, 9)

overtime = st.selectbox("OverTime", ["No", "Yes"])

percent_salary_hike = st.number_input("Percent Salary Hike", 11, 25)

performance_rating = st.number_input("Performance Rating", 3, 4)

relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4)

stock_option_level = st.number_input("Stock Option Level", 0, 3)

total_working_years = st.number_input("Total Working Years", 0, 40)

training_times_last_year = st.number_input("Training Times Last Year", 0, 6)

work_life_balance = st.slider("Work Life Balance", 1, 4)

years_at_company = st.number_input("Years At Company", 0, 40)

years_in_current_role = st.number_input("Years In Current Role", 0, 18)

years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15)

years_with_current_manager = st.number_input("Years With Current Manager", 0, 17)

# Encoding categorical data
business_travel_map = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}

marital_status_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

overtime_map = {
    "No": 0,
    "Yes": 1
}

# Predict button
if st.button("Predict"):#when button is clicked, it return true and code run inside the if block.
    #numpy 2D array,  use for sklearn model prediction and Scalar. np.arrray([[]]) is syntax for 2D array.
    input_data = np.array([[
        age,
        business_travel_map[business_travel], #encode (return text from dropdown list) into numerical values.
        daily_rate,
        distance_from_home,
        education,
        environment_satisfaction,
        hourly_rate,
        job_involvement,
        job_level,
        job_satisfaction,
        marital_status_map[marital_status],#encode (return text from dropdown list) into numerical values.
        monthly_income,
        monthly_rate,
        num_companies_worked,
        overtime_map[overtime],#encode (return text from dropdown list) into numerical values.
        percent_salary_hike,
        performance_rating,
        relationship_satisfaction,
        stock_option_level,
        total_working_years,
        training_times_last_year,
        work_life_balance,
        years_at_company,
        years_in_current_role,
        years_since_last_promotion,
        years_with_current_manager
    ]])

    # Scale the input data before prediction
    input_data_scaled = scaler.transform(input_data)
    #calculate prediction and return results in 1D array. This model is already trained and predict directly class.
    prediction = model.predict(input_data_scaled)
    #calculate probability and return 2D Array probability = [probability for Attrition No, Probability for Attrition Yes].
    #Access Probability of Attrition No = probability[0][0]
    #Access Probability of Attrition No = probability[0][1]
    #Calculation of probability perform for Displaying on Web page.
    probability = model.predict_proba(input_data_scaled)
    #create a medium size heading.
    st.subheader("Prediction Result")
    #condition use to convert 0/1 (model class prediction) into text leave/stay.
    if prediction[0] == 1:
        st.error("Employee will leave the company.") # error used to display in red
    else:
        st.success("Employee will stay in the company.") # success used to diplay in green
    #st.write() show text on web page, and probability converted into persontage and show 2 floating digits (:.2f) after point.
    st.write(f"Leave Probability: {probability[0][1] * 100:.2f}%")
    st.write(f"Stay Probability : {probability[0][0] * 100: .2f}%")