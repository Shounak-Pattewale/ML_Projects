import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# Import Database 
df = pd.read_csv("diabetes.csv")

# Check if null values are present. If there are, fill those null values
# print(df.info())
print(f"\nChecking for null values : {df.info()} \n")

# print(df.head())
# Exploratory Data Analysis

# Histograms for all columns

plt.title("Histogram 1 : Pregnancies")
plt.ylabel("Total Count")
plt.xlabel("Pregnancies")
plt.hist(df['Pregnancies'], width = 1)
plt.show()
plt.clf()

# plt.title("Histogram 2 : Glucose")
# plt.ylabel("Total Count")
# plt.xlabel("Glucose")
# plt.hist(df["Glucose"], width = 10, color = "blue")
# plt.show()
# plt.clf()

# plt.title("Histogram 3 : Blood Pressure")
# plt.ylabel("Total Count")
# plt.xlabel("Blood Pressure")
# plt.hist(df["BloodPressure"], width = 7, color="green")
# plt.show()
# plt.clf()

# plt.title("Histogram 4 : Skin Thickness", color="black")
# plt.ylabel("Total Count")
# plt.xlabel("Skin Thickness")
# plt.hist(df["SkinThickness"], width = 5, color="purple")
# plt.show()
# plt.clf()

# plt.title("Histogram 5 : Insulin")
# plt.ylabel("Total Count")
# plt.xlabel("Insulin")
# plt.hist(df["Insulin"], width = 50)
# plt.show()
# plt.clf()

# plt.title("Histogram 6 : BMI")
# plt.ylabel("Total Count")
# plt.xlabel("BMI")
# plt.hist(df["BMI"], width = 4, color="blue")
# plt.show()
# plt.clf()

# plt.title("Histogram 7 : Diabetes Pedigree Function")
# plt.ylabel("Total Count")
# plt.xlabel("Diabetes Pedigree Function")
# plt.hist(df["DiabetesPedigreeFunction"], width = 0.15, color="green" )
# plt.show()
# plt.clf()

# plt.title("Histogram 8 : Age")
# plt.ylabel("Total Count")
# plt.xlabel("Age")
# plt.hist(df["Age"], width = 4, color="purple")
# plt.show()
# plt.clf()

# plt.title("Histogram 9 : Outcome")
# plt.ylabel("Total Count")
# plt.xlabel("Outcome")
# plt.hist(df["Outcome"], width = 0.1, color="black")
# plt.show()
# plt.clf()

# Scatter Plots

plt.title("Age Vs Pregnancies")
plt.xlabel("Pregnancies")
plt.ylabel("Age")
plt.scatter(df['Pregnancies'],df['Age'])
plt.show()
plt.clf()

# plt.title("Age Vs Glucose")
# plt.xlabel("Glucose")
# plt.ylabel("Age")
# plt.scatter(df['Glucose'],df['Age'])
# plt.show()
# plt.clf()

# plt.title("Age Vs Skin Thickness")
# plt.xlabel("Skin Thickness")
# plt.ylabel("Age")
# plt.scatter(df['SkinThickness'],df['Age'])
# plt.show()
# plt.clf()

# plt.title("Age Vs Insulin")
# plt.xlabel("Insulin")
# plt.ylabel("Age")
# plt.scatter(df['Insulin'],df['Age'])
# plt.show()
# plt.clf()

# plt.title("Age Vs BMI")
# plt.xlabel("BMI")
# plt.ylabel("Age")
# plt.scatter(df['BMI'],df['Age'])
# plt.show()
# plt.clf()

# Modelling

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Split the dataset using “train-test-split” function.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42, stratify = y)

# Apply KNN classification on “Outcome” column of the dataset. Select the appropriate features
# Check score

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn_fit = knn.fit(X_train,y_train)
print(f'\n{knn_fit}')
knn_testscore = knn.score(X_test,y_test)
print(f'\nKNN Testing Score : {knn_testscore}')
knn_trainscore = knn.score(X_train,y_train)
print(f'KNN Training Score : {knn_trainscore}\n')

# Apply Decision Tree Classifier  on “Outcome” column of the dataset. Select the appropriate features
# Check score

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model_fit = model.fit(X_train, y_train)
print(f'\n{model_fit}')
model_testscore = model.score(X_test,y_test)
print(f'\nDecision Tree Testing Score : {model_testscore}')
model_trainscore = model.score(X_train,y_train)
print(f'Decision Tree Training Score : {model_trainscore}\n')

# Predict on new data set (minimum 1 row)
# Here I'm using KNN Classification for prediction

new_df = X_test[0:5]
print(f'New DataFrame containing 5 rows \n\n{new_df}\n')
predict = knn.predict(new_df)
print(f'Prediction based on the above rows : {predict}\n')
(unique, counts) = np.unique(predict, return_counts=True)
freq = np.asarray((unique, counts)).T
print(f'Frequency of each prediction in given set\n{freq}\n')