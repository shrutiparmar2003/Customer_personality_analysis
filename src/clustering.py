import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed_data.csv")

#Selecting features for clustering
features=['Income','Total_Spending','Total_Purchases','Recency','Age','Customer_Tenure']
X=df[features]

#Normalizing Data
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#elbow method for finding optimal number  of clusters in Kmeans 
inertia=[]
K_range=range(2,10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#plot elbow method
plt.figure(figsize=(8,5))
plt.plot(K_range,inertia,marker='o',color='blue')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

#Applying Kmeans with optimal number of clusters that is 4
kmeans=KMeans(n_clusters=4,random_state=42)
df['Cluster']=kmeans.fit_predict(X_scaled)

# Save clustered data
df.to_csv(r"D:\customer_personality_analysis\data\clustered_data.csv", index=False)
print("✅ Clustering completed and saved as clustered_data.csv")
