from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#a function that takes already clean data and returns sc 
def k_mean(df, k):

  scaler = StandardScaler()
  scaler_df = df.copy()

  # : preserves dataframe struct
  scaler_df[:] = scaler.fit_transform(scaler_df)

  kmeans = KMeans(n_clusters=k,random_state=42)
  kmeans.fit(scaler_df)

  scaler_df['Clusters'] = kmeans.labels_

  centroids_scaled = kmeans.cluster_centers_

  centroids_real = scaler.inverse_transform(centroids_scaled)

  return centroids_real


