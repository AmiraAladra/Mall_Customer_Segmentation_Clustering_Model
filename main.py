
from src.data.data_processing import load_data
# from src.features.features_eng import create_new_variables
# from src.features.features_eng import Create_dummy_variables
from src.models.train_model import train_kmeans
from src.visualization.visualize import plot_clusters,elbow_method, silhouette_analysis
import pickle

if __name__ == "__main__":
    try:
        # Load data
        data_path = "data/raw/mall_customers.csv"
        df = load_data(data_path)

        # Features for 2D clustering
        features_2d = ['Annual_Income', 'Spending_Score']
        df_2d, _ = train_kmeans(df, features_2d)
        df_2d, model_2d = train_kmeans(df.copy(), features_2d)
        plot_clusters(df_2d, features_2d, 'reports/figures/kmeans_scatterplot.png')

        # Elbow method
        wss_df = elbow_method(df, features_2d, 'reports/figures/kmeans_Elbow.png')

        # Silhouette method for 2D
        sil_df_2d, best_k_2d = silhouette_analysis(df, features_2d, 'reports/figures/kmeans_Silhouette.png')

        # Silhouette method for 3D clustering
        features_3d = ['Age', 'Annual_Income', 'Spending_Score']
        sil_df_3d, best_k_3d = silhouette_analysis(df, features_3d, 'reports/figures/kmeans_Silhouette_3.png')
        
        # Save clustered data
        df_2d.to_csv("data/processed/mall_customers_clustered.csv", index=False)
        
        with open("models/kmeans_model_2d.pkl", "wb") as f:
            pickle.dump(model_2d, f)

        print(f"Best number of clusters (2D features): {best_k_2d}")
        print(f"Best number of clusters (3D features): {best_k_3d}")

    except Exception as e:
        print(f"An error occurred: {e}")
    