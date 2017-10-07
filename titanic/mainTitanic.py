import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Imputing the age 
def age_imputing(df): 
    # Impute the means of Mr and Mrs to missing data
    missMean = df.Age[df.Name.str.contains("Miss")].mean()
    mrsMean = df.Age[df.Name.str.contains("Mrs")].mean()
    
    df.Age[(df.Name.str.contains("Miss")) & (df['Age'].isnull())] = missMean
    df.Age[(df.Name.str.contains("Mrs")) & (df['Age'].isnull())] = mrsMean
    return df

def normal_cleaning(df, cv):
    # Age Imputing
    before = df.Age.isnull().sum()
    df = age_imputing(df)
    after = df.Age.isnull().sum()
    print("Age imputing solves this many entries:" + str(before-after))
    
    dropped = df.fillna(0)
    
    # Binary encoding of gender
    sex = {'male':1, 'female':0}
    dropped['Sex'] = dropped['Sex'].map(sex)
    print(dropped)
    # Keep age and fare for now
    if (cv):
            dropped = dropped[["PassengerId","Age", "Fare", "Pclass","Parch", "Sex"]]
    else:
        dropped = dropped[["PassengerId","Age", "Fare", "Pclass", "Survived","Parch", "Sex"]]
    return dropped

def random_forest_model(training_data, validation_data, testing_data):
    cols_to_delete = ["Survived", "PassengerId"]
    clf = RandomForestClassifier(max_depth=None,\
                                 random_state=0,\
                                  n_estimators=100)
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


cleaned_training_data = normal_cleaning(training_data, False)
cleaned_validation_data = normal_cleaning(validation_data, False)
cleaned_testing_data = normal_cleaning(testing_data, True)

result = random_forest_model(cleaned_training_data, cleaned_validation_data, \
        cleaned_testing_data)


result.to_csv("result.csv", index=False)

