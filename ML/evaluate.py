# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Đọc dữ liệu
# df = pd.read_csv('train.csv')
# X = df.drop('price_range', axis=1)

# # Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# def evaluate_kmeans(X, max_k=10):
#     # Tính toán các metrics
#     silhouette_scores = []
#     calinski_scores = []
#     davies_scores = []
#     distortions = []
#     K = range(2, max_k+1)
    
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         labels = kmeans.fit_predict(X)
        
#         # Tính các chỉ số
#         silhouette = silhouette_score(X, labels)
#         calinski = calinski_harabasz_score(X, labels)
#         davies = davies_bouldin_score(X, labels)
        
#         silhouette_scores.append(silhouette)
#         calinski_scores.append(calinski)
#         davies_scores.append(davies)
#         distortions.append(kmeans.inertia_)
        
#         print(f'K={k}:')
#         print(f'Silhouette Score: {silhouette:.3f}')
#         print(f'Calinski-Harabasz Score: {calinski:.3f}')
#         print(f'Davies-Bouldin Score: {davies:.3f}')
#         print(f'Inertia: {kmeans.inertia_:.3f}\n')
    
#     # Vẽ đồ thị kết quả
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
    
#     ax1.plot(K, distortions, 'bx-')
#     ax1.set_xlabel('Số cụm (k)')
#     ax1.set_ylabel('Độ méo mó (Distortion)')
#     ax1.set_title('Phương pháp Elbow')
    
#     ax2.plot(K, silhouette_scores, 'rx-')
#     ax2.set_xlabel('Số cụm (k)')
#     ax2.set_ylabel('Điểm Silhouette')
#     ax2.set_title('Phân tích Silhouette')
    
#     ax3.plot(K, calinski_scores, 'gx-')
#     ax3.set_xlabel('Số cụm (k)')
#     ax3.set_ylabel('Điểm Calinski-Harabasz')
#     ax3.set_title('Phân tích Calinski-Harabasz')
    
#     ax4.plot(K, davies_scores, 'yx-')
#     ax4.set_xlabel('Số cụm (k)')
#     ax4.set_ylabel('Điểm Davies-Bouldin')
#     ax4.set_title('Phân tích Davies-Bouldin')
    
#     plt.tight_layout()
#     plt.show()
    
#     return silhouette_scores, calinski_scores, davies_scores, distortions

# # Đánh giá mô hình
# scores = evaluate_kmeans(X_scaled, max_k=10)

# # Phân tích chi tiết với số cụm tối ưu (giả sử k=4)
# k_optimal = 4
# kmeans = KMeans(n_clusters=k_optimal, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)

# # Thêm nhãn cụm vào DataFrame
# df_with_clusters = df.copy()
# df_with_clusters['Cluster'] = clusters

# # Phân tích đặc điểm của từng cụm
# print("\nĐặc điểm của từng cụm:")
# for cluster in range(k_optimal):
#     print(f"\nCụm {cluster}:")
#     cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
#     print(f"Số lượng điện thoại: {len(cluster_data)}")
#     print("\nGiá trị trung bình của các đặc trưng:")
#     print(cluster_data.mean())
    
#     # Phân phối price_range trong cụm
#     print("\nPhân phối price_range:")
#     print(cluster_data['price_range'].value_counts().sort_index())

# # Visualize phân phối price_range trong các cụm
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Cluster', y='price_range', data=df_with_clusters)
# plt.title('Phân phối Price Range trong các cụm')
# plt.xlabel('Cụm')
# plt.ylabel('Price Range')
# plt.show()

# # Visualize các đặc trưng quan trọng
# important_features = ['battery_power', 'ram', 'px_width', 'px_height', 'mobile_wt']
# fig, axes = plt.subplots(len(important_features), 1, figsize=(12, 4*len(important_features)))
# fig.suptitle('Phân phối các đặc trưng quan trọng trong các cụm')

# for i, feature in enumerate(important_features):
#     sns.boxplot(x='Cluster', y=feature, data=df_with_clusters, ax=axes[i])
#     axes[i].set_title(feature)

# plt.tight_layout()
# plt.show()

# # Visualize mối quan hệ giữa các đặc trưng
# plt.figure(figsize=(12, 8))
# sns.scatterplot(data=df_with_clusters, x='ram', y='battery_power', hue='Cluster', palette='deep')
# plt.title('Mối quan hệ giữa RAM và Battery Power trong các cụm')
# plt.show()

# # Tính toán correlation matrix
# correlation = df_with_clusters.corr()

# # Visualize correlation matrix
# plt.figure(figsize=(15, 10))
# sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
# plt.title('Ma trận tương quan giữa các đặc trưng')
# plt.show()
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('train.csv')
X = df.drop('price_range', axis=1)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Thiết lập số cụm tối đa cần kiểm tra
num_clusters = 10
b = []

# Thử nghiệm với các số cụm khác nhau
for i in range(2, num_clusters + 1):
   # Thực hiện phân cụm K-means
   kmeans = KMeans(n_clusters=i, random_state=42)
   kmeans.fit(X_scaled)
   labels = kmeans.labels_
   
   # Tính các chỉ số đánh giá
   silhouette = silhouette_score(X_scaled, labels)
   davies_bouldin = davies_bouldin_score(X_scaled, labels)
   
   # Lưu kết quả
   new_list = [silhouette, davies_bouldin, i]
   b.append(new_list)
   
   print(f"Số cụm {i}:")
   print(f"Silhouette Score: {silhouette:.3f}")
   print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
   print("----------------------------------------")

# Tìm kết quả tối ưu
max_b = max(b)
print("\nKết quả tối ưu:")
print(f"Silhouette Score: {max_b[0]:.3f}")
print(f"Davies-Bouldin Score: {max_b[1]:.3f}") 
print(f"Số cụm tối ưu: {max_b[2]}")

# Vẽ đồ thị Elbow Method
distortions = []
K = range(2, num_clusters + 1)
for k in K:
   kmeans = KMeans(n_clusters=k, random_state=42)
   kmeans.fit(X_scaled)
   distortions.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()

# Vẽ đồ thị Silhouette Score
silhouette_scores = [score[0] for score in b]
plt.figure(figsize=(10,6))
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()

# Vẽ đồ thị Davies-Bouldin Score
davies_scores = [score[1] for score in b]
plt.figure(figsize=(10,6))
plt.plot(K, davies_scores, 'gx-')
plt.xlabel('k')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Analysis')
plt.show()

# Phân tích chi tiết với số cụm tối ưu
optimal_k = max_b[2]
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
labels_optimal = kmeans_optimal.fit_predict(X_scaled)

# Thêm nhãn cụm vào DataFrame
df_with_clusters = df.copy()
df_with_clusters['Cluster'] = labels_optimal

# Phân tích đặc điểm của từng cụm
print("\nĐặc điểm của từng cụm:")
for cluster in range(optimal_k):
   print(f"\nCụm {cluster}:")
   cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
   print(f"Số lượng điện thoại: {len(cluster_data)}")
   print("\nGiá trị trung bình của các đặc trưng:")
   print(cluster_data.mean())
   print("\nPhân phối price_range:")
   print(cluster_data['price_range'].value_counts().sort_index())

# Visualize phân phối price_range trong các cụm
plt.figure(figsize=(10,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_optimal, cmap='viridis')
plt.title('Phân cụm K-means')
plt.colorbar()
plt.show()

# Visualize phân phối price_range trong các cụm
plt.figure(figsize=(12, 6))
for cluster in range(optimal_k):
   cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
   plt.hist(cluster_data['price_range'], alpha=0.5, label=f'Cluster {cluster}')
plt.title('Phân phối Price Range trong các cụm')
plt.xlabel('Price Range')
plt.ylabel('Số lượng')
plt.legend()
plt.show()