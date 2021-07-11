import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# load the Random Forest Classifier
with open("./rfSimple.pkl", 'rb') as file:
    rf = pickle.load(file)

# the model takes the following variables as input:
    # Age - int
    # child - int
    # family_size - int
    # Pclass_1 - uint8
    # Pclass_2 - uint8
    # Pclass_3 - uint8
    # Sex_female - uint8
    # Sex_male - uint8
    # Embarked_C - uint8
    # Embarked_Q - uint8
    # Embarked_S - uint8

# create model input
Age = st.sidebar.number_input("How old are you?", 0, 100, 30)
child = int(Age <= 16)
family_size = st.sidebar.number_input("How many family members are aboard the ship (including yourself)?", 1, 20, 1)
pclassAux = st.sidebar.selectbox("In which passenger class are you traveling?", (1,2,3))
Pclass_1=0
Pclass_2=0
Pclass_3=0
if pclassAux==1:
    Pclass_1=1
if pclassAux==2:
    Pclass_2=1
if pclassAux==3:
    Pclass_3=1
sex = st.sidebar.selectbox("Are you male or female?", ("male", "female"), index=1)
Sex_female = 0
Sex_male = 0
if sex=="female":
    Sex_female=1
else:
    Sex_male=1
embarked = st.sidebar.selectbox("Which is your port of Embarkation?", ("Cherbourg", "Queenstown", "Southampton"))
Embarked_C = 0
Embarked_Q = 0
Embarked_S = 0

# create input DataFrame
inputDF = pd.DataFrame({"Age": Age,
                        "child": child,
                        "family_size": family_size,
                        "Pclass_1": Pclass_1,
                        "Pclass_2": Pclass_2,
                        "Pclass_3": Pclass_3,
                        "Sex_female": Sex_female,
                        "Sex_male": Sex_male,
                        "Embarked_C": Embarked_C,
                        "Embarked_Q": Embarked_Q,
                        "Embarked_S": Embarked_S},
                                 index=[0])

SurvivalProba = rf.predict_proba(inputDF)[0,1]
survPerc = round(SurvivalProba*100, 1)

# display survival probability
st.image("./static/titanic.jpg", use_column_width=True)
st.write("Your Survival Probability based on the information provided is: {}%.".format(survPerc))
