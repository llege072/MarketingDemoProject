from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd

class k_mean_model():
  def __init__(self, csv_file, k):
    
    self.orig_df = pd.read_csv(csv_file)
    self.k = k
    self.cleaned_df = None
    self.scaled_df = None

  #df is expected to only have the columns it wants
  def set_cleaned_df(self,df):
    self.cleaned_df = df




#a function that takes already clean data and returns sc 
def k_mean(df, k):

  scaler = StandardScaler()
  scaler_df = df.copy()

  # : preserves dataframe struct
  scaler_df[:] = scaler.fit_transform(scaler_df)

  kmeans = KMeans(n_clusters=k,random_state=42)
  kmeans.fit(scaler_df)

  df['Clusters'] = kmeans.labels_

  centroids_scaled = kmeans.cluster_centers_

  centroids_real = scaler.inverse_transform(centroids_scaled)

  return centroids_real

#draws graph to show where the clusters are (ChatGPT generated)
def draw_kmean_plot(df, centroids, attr1, attr2):
  # Plot Income vs Spending Score and color by cluster
  plt.figure(figsize=(8,6))

  # Scatter plot of all customers 
  plt.scatter(df[attr1], df[attr2],
            c=df['Clusters'], cmap='viridis', s=50, alpha=0.7)

  # Plot centroids 
  plt.scatter(centroids[:,1], centroids[:,2],
            c='red', marker='X', s=200, label='Centroids')
  
  # Labels and title
  plt.xlabel("Annual Income (k$)")  
  plt.ylabel("Spending Score (1-100)")
  plt.title("Customer Segments: Income vs Spending Score")
  plt.legend()
  plt.show()

  





