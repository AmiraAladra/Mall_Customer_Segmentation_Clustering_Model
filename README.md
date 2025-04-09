# Customer Segmentation with KMeans
## Project Overview
This project uses KMeans Clustering to segment mall customers into meaningful groups based on their income, spending habits, and age. It helps identify customer personas and allows for targeted marketing strategies.

The app provides an interactive experience where users can input their own information and discover which cluster they belong to, compare themselves to similar customers, and visually explore clustering results.

## 👉 Live App:
https://mall-customer-segmentation-clustering-model.streamlit.app/


## 🚀 Features
Interactive Prediction: Users input Age, Annual Income, and Spending Score to discover their customer cluster.

Cluster Visualization: Shows how customers are distributed across clusters with a scatterplot.

Personalized Insights: Compares user input to others in the same cluster and gives an interpretation.

Model Evaluation: Elbow and Silhouette plots are used to determine optimal clusters.

Error Handling: Handles missing model/data/figure files gracefully with user-friendly messages.

## 📦 Dataset
The dataset used for training is mall_customers.csv, containing the following features:

Column Name	Description
CustomerID	Unique customer ID
Gender	Customer gender
Age	Customer's age
Annual_Income	Annual income in $k
Spending_Score	Score assigned by the mall based on spending behavior
🛠 Technologies Used
Python 3.x

## 📚 Libraries:
pandas: Data handling and manipulation

scikit-learn: Clustering and model evaluation

matplotlib, seaborn: Visualization

pickle: Saving/loading trained models

Streamlit: Web app interface

os: Directory handling and checks

## 🔍 Code Explanation
load_data(data_path): Loads CSV, handles missing data with error handling.

train_kmeans(df, features): Trains a KMeans clustering model and assigns clusters.

plot_clusters(df, features, path): Saves cluster scatterplot.

elbow_method(df, features, path): Saves elbow plot using WCSS (Within Cluster Sum of Squares).

silhouette_analysis(df, features, path): Saves silhouette score plot to evaluate clustering quality.

main(): The entry point for training, evaluation, and saving the model and clustered dataset.

## 🌐 Streamlit App Features
Sidebar form to enter:

Age

Annual Income

Spending Score

After clicking Predict:

Shows which cluster the user belongs to

Summary stats of similar customers (Mean, Min, Max)

Human-readable interpretation of the cluster

A message highlighting what traits user shares with others

A scatterplot showing all clusters and the user's position

Static visualizations:

Cluster Plot

Elbow Plot

Silhouette Plots

## 📁 Project Structure
        .
        ├── data/
        │   ├── raw/                         # Raw dataset (mall_customers.csv)
        │   └── processed/                   # Clustered dataset (mall_customers_clustered.csv)
        ├── models/                          # Trained model (kmeans_model_2d.pkl)
        ├── reports/
        │   └── figures/                     # Generated plots
        │       ├── kmeans_scatterplot.png
        │       ├── kmeans_Elbow.png
        │       ├── kmeans_Silhouette.png
        │       └── kmeans_Silhouette_3.png
        ├── src/
        │   ├── data/                        # load_data()
        │   ├── models/                      # train_kmeans()
        │   ├── visualization/              # plotting functions
        │   └── main.py                     # Main training script
        ├── app.py                          # Streamlit web app
        ├── requirements.txt                # Python dependencies
        └── README.md                       # Project documentation
        
## 🖥 Installation (For Local Deployment)
1. Clone the Repository

git clone https://github.com/yourusername/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
2. Install Dependencies
pip install -r requirements.txt
3. Train Model & Generate Plots
python src/main.py
This will:

Train the model

Save it to models/kmeans_model_2d.pkl

Save cluster assignments to data/processed/mall_customers_clustered.csv

Generate Elbow and Silhouette plots

4. Run the App
streamlit run app.py


## 📈 Output
models/kmeans_model_2d.pkl: Trained KMeans model

data/processed/mall_customers_clustered.csv: Customer data with cluster labels

reports/figures/: Visualizations for analysis and interpretation

## 🙌 Thank You!
Thank you for exploring the Customer Segmentation App!
Feel free to contribute, raise issues, or share your feedback.
This tool is great for marketing teams, data analysts, and students learning clustering.
