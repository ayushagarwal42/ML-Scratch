
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.Classification.LogisticRegression import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.Classification.KNN import KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:/Users/This PC/Desktop/Machine Learning/ML-Scratch/Data Folder/diabetes.csv")

df.head()

df.isna().sum()

df.info()

df.describe()

zeros_count = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
print(zeros_count)

columnswithzero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for column in columnswithzero:
    mean_value = df[column].mean()
    df[column] = df[column].replace(0, mean_value)

df.describe()

outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)

plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Outcome')

sns.countplot(data=df, x=df['Outcome'])

df.hist(figsize=(20, 20))

# columns_to_plot = [col for col in df.columns if col != 'Outcome']
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

# Loop through the columns
for col in columns:
    plt.figure()
    sns.kdeplot(data=df, x=col, hue='Outcome', fill=True)
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.title(f'KDE Plot of {col} by Outcome')
    plt.show()



corrmat = df.corr()
print(corrmat)

sns.heatmap(corrmat, annot=True, cmap="RdYlGn")

# X=df.drop('Outcome','Pregnancies','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction',axis=1)
X = df[['Glucose', 'BMI', 'Age']]
y = df['Outcome']
print(X)
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

## Logistic regression
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
#### logistic predictions
lgr_pred = lgr.predict(X_test)
#### logistic accuracy
lgr_acc = accuracy_score(y_test, lgr_pred)
print(lgr_acc)

print("Classification Report:")
print(classification_report(y_test, lgr_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, lgr_pred))

# Logistic from Scratch
lgr = LogisticRegression(lr=0.01)
lgr.fit(X_train, y_train)
lgr_pred = lgr.predict(X_test)
lgr_acc = accuracy_score(y_test, lgr_pred)
print("scratch lgr ", lgr_acc)

## KNN
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
#### KNN prediction
knn_pred = knn.predict(X_test)
#### KNN accuracy
knn_acc = accuracy_score(y_test, knn_pred)
print(knn_acc)

scores = []
for i in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, knn_pred))

plt.plot(range(1, 25), scores)

### Knn from scratch
### To use Euclidean distance:
knn_euclidean = KNN(k=13, distance_metric='euclidean')
knn_euclidean.fit(X_train, y_train)
knn_euclidean_pred = knn_euclidean.predict(X_test)

knn_acc = accuracy_score(y_test, knn_euclidean_pred)
print(knn_acc)

### To use Manhattan distance:
knn_manhattan = KNN(k=13, distance_metric='manhattan')
knn_manhattan.fit(X_train, y_train)
knn_manhattan_pred = knn_manhattan.predict(X_test)

knn_acc = accuracy_score(y_test, knn_manhattan_pred)
print(knn_acc)

## Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
#### naive bayes prediction
nb_pred = nb.predict(X_test)
#### naive bayes accuracy
nb_acc = accuracy_score(y_test, nb_pred)
print(nb_acc)

## Support vector machine
sv = SVC()
sv.fit(X_train, y_train)
#### svm prediction
sv_pred = sv.predict(X_test)
#### svm accuracy
sv_acc = accuracy_score(y_test, sv_pred)
print(sv_acc)

## Decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
#### decision tree prediction
dt_pred = dt.predict(X_test)
#### decision tree accuracy
dt_acc = accuracy_score(y_test, dt_pred)
print(dt_acc)

## Random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
#### random forest prediction
rf_pred = rf.predict(X_test)
#### random forest accuracy
rf_acc = accuracy_score(y_test, rf_pred)
print(rf_acc)