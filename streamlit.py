import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# --- Load Model and Data ---
try:
    with open("models/kmeans_model_2d.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Trained model not found! Make sure 'models/kmeans_model_2d.pkl' exists.")
    st.stop()

try:
    df_clustered = pd.read_csv("data/processed/mall_customers_clustered.csv")
except FileNotFoundError:
    st.error("❌ Clustered dataset not found! Make sure 'mall_customers_clustered.csv' exists.")
    st.stop()

# --- UI ---
st.title("🧩 Customer Segmentation with KMeans")

with st.sidebar:
    st.header("📥 Enter Customer Details")
    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Annual Income (k$)", 15, 150, 60)
    score = st.slider("Spending Score", 1, 100, 50)

    predict_btn = st.button("🔍 Predict Cluster")

# --- Prediction Section ---
if predict_btn:
    try:
        user_input = pd.DataFrame([[income, score]], columns=["Annual_Income", "Spending_Score"])
        cluster = model.predict(user_input)[0]
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.stop()

    st.subheader(f"🎯 Predicted Cluster: {cluster}")

    cluster_data = df_clustered[df_clustered["Cluster"] == cluster]

    # Show summary stats (mean, min, max)
    stats = cluster_data[["Age", "Annual_Income", "Spending_Score"]].agg(["mean", "min", "max"]).T
    stats = stats.rename(columns={"mean": "Mean", "min": "Min", "max": "Max"})
    st.markdown("### 📊 Cluster Statistics (Mean, Min, Max)")
    st.dataframe(stats)

    # --- Interpretation ---
    st.markdown(f"""### 🧠 Cluster Interpretation  
Your input (**Age**: {age}, **Income**: {income}, **Score**: {score}) places you in **Cluster {cluster}**,  
which represents customers with:
- 👤 **Average Age:** {stats.loc['Age', 'Mean']:.1f} years  
- 💰 **Average Income:** ${stats.loc['Annual_Income', 'Mean']:.1f}k  
- 🛍️ **Average Spending Score:** {stats.loc['Spending_Score', 'Mean']:.1f}  
""")

    # --- Similarity Message ---
    similarity_msg = "🧩 You are most similar to customers who"
    if abs(age - stats.loc['Age', 'Mean']) < 5:
        similarity_msg += f" are around your age ({age})"
    if abs(income - stats.loc['Annual_Income', 'Mean']) < 10:
        similarity_msg += f", earn a similar income (~${income}k)"
    if abs(score - stats.loc['Spending_Score', 'Mean']) < 10:
        similarity_msg += f", and have similar spending habits ({score})."
    similarity_msg = similarity_msg.strip(", ") + "."

    st.markdown(f"### 💡 {similarity_msg}")

    # --- Cluster Scatter Plot ---
    fig, ax = plt.subplots()
    for c in df_clustered["Cluster"].unique():
        cluster_subset = df_clustered[df_clustered["Cluster"] == c]
        ax.scatter(cluster_subset["Annual_Income"], cluster_subset["Spending_Score"], label=f"Cluster {c}", alpha=0.5)
    ax.scatter(income, score, color="red", s=100, label="You", edgecolor="black")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("🖼️ Cluster Visualization with Your Input")
    ax.legend()
    st.pyplot(fig)

# --- Model Explanation and Visualizations ---
st.markdown("---")
st.header("📘 About the Clustering Model")

st.markdown("""
**KMeans Clustering** is an unsupervised machine learning algorithm used to group similar data points into **clusters**.

In this app:
- We used customer **Annual Income** and **Spending Score** to segment them.
- The model tries to find groupings (clusters) of customers that share similar behavior.
- We evaluated the clustering performance using the **Elbow Method** and **Silhouette Score**.

These figures were generated during training:
""")

fig_paths = {
    "📌 KMeans Cluster Plot": "reports/figures/kmeans_scatterplot.png",
    "📈 Elbow Plot": "reports/figures/kmeans_Elbow.png",
    "📊 Silhouette Plot (2D)": "reports/figures/kmeans_Silhouette.png",
    "🔺 Silhouette Plot (3D)": "reports/figures/kmeans_Silhouette_3.png"
}

for title, path in fig_paths.items():
    if os.path.exists(path):
        st.subheader(title)
        st.image(path, use_column_width=True)
    else:
        st.warning(f"⚠️ Could not find image: `{path}`")
