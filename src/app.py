
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.Classification.KNN import KNN
from src.Classification.LogisticRegression import LogisticRegression
from src.Classification.SVM import SVM
from src.Clustering.DBSCAN import DBSCAN
from src.Clustering.MeanShiftClustering import MeanShift
from src.Clustering.kmeans import KMeans
from sklearn.metrics import silhouette_score



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

## Kmeans scratch
inertia_values = []
for k in range(1, 7):
    kmeans = KMeans(K=k)
    kmeans.predict(X_scaled)
    inertia_values.append(kmeans.inertia)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 7), inertia_values, linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

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

kmeans_silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
print(f"kmeans_Silhouette Score: {kmeans_silhouette_avg}")  # 0.321

## mean shift scratch
mean_shift = MeanShift(radius=2.5)
mean_shift.fit(X_scaled)
labels = mean_shift.predict(X_scaled)
centroids = mean_shift.centroids
print(np.unique(labels))
print(centroids)

mean_shift_silhouette_avg = silhouette_score(X_scaled, labels)
print(f"mean_shift_Silhouette Score: {mean_shift_silhouette_avg}")  # 0.250

# Assuming X has 3 features
feature1 = X_scaled[:, 0]
feature2 = X_scaled[:, 1]
feature3 = X_scaled[:, 2]

# Create a 3D scatter plot to visualize the clusters
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop through each unique cluster label and plot the data points for that cluster
for label in np.unique(labels):
    cluster = X_scaled[labels == label]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f'Cluster {label}')
# Plot the centroids
for i, centroid in centroids.items():
    ax.scatter(centroid[0], centroid[1], centroid[2], s=100, c='black', marker='X', label=f'Centroid {i}')

ax.set_title('MeanShift Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()

# dbscan
dbscan = DBSCAN(eps=0.2, min_samples=5)
cluster_labels = dbscan.fit(X_scaled)

# Compute Silhouette Score
dbscan_silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"dbscan_Silhouette Score: {dbscan_silhouette_avg}")  # -0.284

# Visualize the DBSCAN clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()
