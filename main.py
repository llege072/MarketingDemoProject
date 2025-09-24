from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv("Mall_Customers.csv")

#Drop both IDs
df = df.drop(['CustomerID'], axis=1)

#Convert class to Binary
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})

#Now we must scale the attributes to ensure theres no distortion and all features contribute equally to the distance calculation in k-means
scaler = StandardScaler()
scaler_df = df[['Age','Annual Income (k$)','Spending Score (1-100)']].copy()

scaler_df[:] = scaler.fit_transform(scaler_df) #To preserve DataFrame structure
#---------------------
K = 4

kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(scaler_df)

scaler_df['Cluster'] = kmeans.labels_ #Adds new column

centroids_scaled = kmeans.cluster_centers_

centroids_real = scaler.inverse_transform(centroids_scaled) #Undo the scaling

centroid_df = pd.DataFrame(centroids_real, columns=['Age','Anuual Income (k$)','Spending Score (1-100)'])

print(centroid_df)

# Plot Income vs Spending Score and color by cluster
plt.figure(figsize=(8,6))

# Scatter plot of all customers
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=scaler_df['Cluster'], cmap='viridis', s=50, alpha=0.7)

# Plot centroids (you already computed centroids_real)
plt.scatter(centroids_real[:,1], centroids_real[:,2],
            c='red', marker='X', s=200, label='Centroids')

# Labels and title
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments: Income vs Spending Score")
plt.legend()
plt.show()
