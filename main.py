import pandas as pd
import numpy as np

import k_mean

df = pd.read_csv("Mall_Customers.csv")

#Drop both IDs
df = df.drop(['CustomerID'], axis=1)

#Convert class to Binary
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})

df = df[['Age','Annual Income (k$)','Spending Score (1-100)']]

centroid_df = k_mean.k_mean(df, 4)

k_mean.draw_kmean_plot(df, centroid_df, 'Annual Income (k$)', 'Spending Score (1-100)')