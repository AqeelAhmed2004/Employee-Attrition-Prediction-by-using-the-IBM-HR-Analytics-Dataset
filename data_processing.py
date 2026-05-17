import pandas as pd 
df_pro = pd.read_excel("dataset.xlsx")
#Droping unnecessary features.
#axis=1 means working on columns/axis=0 --> working on rows
#inplace=True means changing dataframe directly no need of assigning.
df_pro.drop(["Department","EducationField","EmployeeCount",
             "EmployeeNumber","JobRole", "Gender",
             "Over18","StandardHours"],axis=1,inplace=True)
#Encoding (Categorical features into numerical features)
#BusinessTravel Feature Encoding
df_pro["BusinessTravel"] = df_pro["BusinessTravel"].map({"Travel_Rarely":1, "Travel_Frequently" : 2, "Non-Travel" : 0})
#MaterialStatus Feature Encoding
df_pro["MaritalStatus"] = df_pro["MaritalStatus"].map({"Single" : 0 ,  "Married" : 1, "Divorced" : 2})
#Attrition Feature Encoding
df_pro["Attrition"] = df_pro["Attrition"].map({"Yes": 1, "No": 0})
#OverTime Feature Encoding
df_pro["OverTime"] = df_pro["OverTime"].map({"Yes": 1, "No": 0})

print(df_pro.head())
