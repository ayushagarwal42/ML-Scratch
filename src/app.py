
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
from src.Classification.SVM import SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.Clustering.kmeans import KMeans

df = pd.read_csv("diabetes.csv")
df.head()
df.isna().sum()
df.info()
df.describe()

zeros_count = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
print(zeros_count)

columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for column in columns_with_zero:
    mean_value = df[column].mean()
    df[column] = df[column].replace(0, mean_value)

df.describe()

outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)

print(plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140))
print(plt.title('Distribution of Outcome'))

print(sns.countplot(data=df, x=df['Outcome']))
print(df.hist())

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

corr_mat = df.corr()
print(corr_mat)

print(sns.heatmap(corr_mat, annot=True, cmap="RdYlGn"))

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
print("Lgr_acc", lgr_acc)

print("Classification Report:")
print(classification_report(y_test, lgr_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, lgr_pred))

# Logistic from Scratch
lgr = LogisticRegression(lr=0.01)
lgr.fit(X_train, y_train)
lgr_pred = lgr.predict(X_test)
lgr_acc = accuracy_score(y_test, lgr_pred)
print("scratch lgr", lgr_acc)

## KNN
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
#### KNN prediction
knn_pred = knn.predict(X_test)
#### KNN accuracy
knn_acc = accuracy_score(y_test, knn_pred)
print("Knn acc", knn_acc)

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
print("Knn_euclidean", knn_acc)

### To use Manhattan distance:
knn_manhattan = KNN(k=13, distance_metric='manhattan')
knn_manhattan.fit(X_train, y_train)
knn_manhattan_pred = knn_manhattan.predict(X_test)

knn_acc = accuracy_score(y_test, knn_manhattan_pred)
print("Knn_manhattan", knn_acc)

## Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
#### naive bayes prediction
nb_pred = nb.predict(X_test)
#### naive bayes accuracy
nb_acc = accuracy_score(y_test, nb_pred)
print("nb_acc", nb_acc)

## Support vector machine
sv = SVC()
sv.fit(X_train, y_train)
#### svm prediction
sv_pred = sv.predict(X_test)
#### svm accuracy
sv_acc = accuracy_score(y_test, sv_pred)
print("svm acc", sv_acc)

# Support vector scratch
clf = SVM()
clf.fit(X_train, y_train)
svs_pred = clf.predict(X_test)
svs_acc = accuracy_score(y_test, svs_pred)
print("scratch svm", svs_acc)

## Decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
#### decision tree prediction
dt_pred = dt.predict(X_test)
#### decision tree accuracy
dt_acc = accuracy_score(y_test, dt_pred)
print("dt_acc", dt_acc)

## Random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
#### random forest prediction
rf_pred = rf.predict(X_test)
#### random forest accuracy
rf_acc = accuracy_score(y_test, rf_pred)
print("rf_acc", rf_acc)

kmeans = KMeans(K=2)
kmeans_labels = kmeans.predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[kmeans_labels == 0, 0], X_scaled[kmeans_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[kmeans_labels == 1, 0], X_scaled[kmeans_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(kmeans.centroids[0][0], kmeans.centroids[0][1], s=300, c='black', marker='X', label='Centroid 1')
plt.scatter(kmeans.centroids[1][0], kmeans.centroids[1][1], s=300, c='black', marker='X', label='Centroid 2')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
