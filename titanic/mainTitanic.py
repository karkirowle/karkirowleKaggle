import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Imputing the age 
def age_imputing(df): 
    # You want an age distribution for Ms and Mrs
    df.Age[df.Name.str.contains("Miss")].plot(kind='kde')    # plots kernel density estimate of ages
    df.Age[df.Name.str.contains("Mrs")].plot(kind='kde')    # plots kernel density estimate of ages
    # plots an axis lable
    plt.xlabel("Age")    
    plt.title("Age Distribution within Mr and Miss")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
    plt.show()

def normal_cleaning(df):
    dropped = df.dropna()
    # Keep age and fare for now
    dropped = dropped[["Age", "Fare", "Pclass", "Survived"]]
    return dropped

def random_forest_model(df, df2):
    cols_to_delete = ["Survived"]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(df.drop(cols_to_delete, axis=1), df["Survived"])
    predicted = clf.predict(df.drop(cols_to_delete, axis=1))
    # Training accuracy
    training_score = accuracy_score(df["Survived"], predicted)
    # Testing accuracy
    predicted_test = clf.predict(df2.drop(cols_to_delete, axis=1))
    testing_score = accuracy_score(df2["Survived"], predicted_test)
    print("Training accuracy " + str(training_score))
    print("Testing accuracy " + str(testing_score))
    

    
# TODO: CV partitioning

input_data = pd.read_csv("data/train.csv")
testing_data = pd.read_csv("data/test.csv")

training_data, validation_data = train_test_split(input_data, test_size=0.2)

# TODO: Come up with a method how age can be predicted
#ageImputing(df)

print(training_data)
cleaned_training_data = normal_cleaning(training_data)
cleaned_validation_data = normal_cleaning(validation_data)

random_forest_model(cleaned_training_data, cleaned_validation_data)
