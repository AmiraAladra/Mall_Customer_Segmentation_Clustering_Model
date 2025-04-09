import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple


def plot_clusters(df, features,figurePath):
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=df, palette='colorblind')
    plt.title("KMeans Clustering Results")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()

def elbow_method(df, features, figurePath):
    
    WCSS = []
    k_values = range(3, 9)
    for k in k_values:
        kmodel = KMeans(n_clusters=k, random_state=42)
        kmodel.fit(df[features])
        WCSS.append(kmodel.inertia_)
    wss_df = pd.DataFrame({'cluster': list(k_values), 'WSS_Score': WCSS})
    wss_df.plot(x='cluster', y='WSS_Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()
    return wss_df

def silhouette_analysis(df, features,figurePath):
    
    silhouette_scores = []
    k_values = range(3, 9)
    for k in k_values:
        kmodel = KMeans(n_clusters=k, random_state=42)
        labels = kmodel.fit_predict(df[features])
        score = silhouette_score(df[features], labels)
        silhouette_scores.append(score)
    sil_df = pd.DataFrame({'cluster': list(k_values), 'Silhouette_Score': silhouette_scores})
    sil_df.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close()
    best_cluster = sil_df.loc[sil_df['Silhouette_Score'].idxmax(), 'cluster']
    return sil_df, best_cluster