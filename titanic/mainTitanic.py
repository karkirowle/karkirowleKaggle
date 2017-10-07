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
    df.Age[df.Name.str.contains("Miss")].plot(kind='kde')    
    df.Age[df.Name.str.contains("Mrs")].plot(kind='kde')    
    # plots an axis lable
    plt.xlabel("Age")    
    plt.title("Age Distribution within Mr and Miss")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
    plt.show()

def normal_cleaning(df):
    dropped = df.fillna(0)
    # Keep age and fare for now
    dropped = dropped[["PassengerId","Age", "Fare", "Pclass", "Survived","Parch"]]
    return dropped
def normal_cleaningCV(df):
    dropped = df.fillna(0)
    # Keep age and fare for now
    dropped = dropped[["PassengerId","Age", "Fare", "Pclass","Parch"]]
    return dropped

def random_forest_model(training_data, validation_data, testing_data):
    cols_to_delete = ["Survived", "PassengerId"]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(training_data.drop(cols_to_delete, axis=1), \
             training_data["Survived"])
    predicted = clf.predict(training_data.drop(cols_to_delete, axis=1))
    # Training accuracy
    training_score = accuracy_score(training_data["Survived"], predicted)
    # Testing accuracy
    predicted_test = clf.predict(validation_data.drop(cols_to_delete, axis=1))
    testing_score = accuracy_score(validation_data["Survived"], predicted_test)
    print("Training accuracy " + str(training_score))
    print("Cross-validation accuracy " + str(testing_score))
    output_table = testing_data[["PassengerId","Age"]]
    output_table['Survived'] = clf.predict(testing_data.drop("PassengerId", \
            axis=1))
    output_table = output_table.drop("Age", axis=1)
    return output_table
    


input_data = pd.read_csv("data/train.csv")
testing_data = pd.read_csv("data/test.csv")

training_data, validation_data = train_test_split(input_data, test_size=0.2)

# TODO: Come up with a method how age can be predicted
#ageImputing(df)


cleaned_training_data = normal_cleaning(training_data)
cleaned_validation_data = normal_cleaning(validation_data)
cleaned_testing_data = normal_cleaningCV(testing_data)

result = random_forest_model(cleaned_training_data, cleaned_validation_data, \
        cleaned_testing_data)

print(result)

result.to_csv("result.csv", index=False)

