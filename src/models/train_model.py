import pandas as pd
from sklearn.cluster import KMeans

def train_kmeans(df, features):
    # Let's train our model on spending_score and annual_income
    kmodel = KMeans(n_clusters= 5, init='k-means++', random_state= 42)
    labels = kmodel.fit_predict(df[features])
    df['Cluster'] = labels
    return df, kmodel