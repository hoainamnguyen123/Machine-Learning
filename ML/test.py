# from tkinter import *
# from tkinter import messagebox
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# import numpy as np
# import pandas as pd

# #đọc dữ liệu
# df = pd.read_csv('train.csv') 
# # Cập nhật danh sách cột được chọn
# selected_columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 
#                    'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 
#                    'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']


# data = np.array(df[selected_columns].values)
# train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)

# #Tạo dictionary ánh xạ tên tiếng Anh sang tiếng Việt
# vietnamese_labels = {
#     'battery_power': 'Dung lượng pin (mAh)',
#     'blue': 'Bluetooth (0/1)',
#     'clock_speed': 'Tốc độ xử lý (GHz)',
#     'dual_sim': 'Hỗ trợ 2 SIM (0/1)',
#     'fc': 'Camera trước (MP)',
#     'four_g': 'Hỗ trợ 4G (0/1)',
#     'int_memory': 'Bộ nhớ trong (GB)',
#     'm_dep': 'Độ dày máy (mm)',
#     'mobile_wt': 'Trọng lượng (g)',
#     'n_cores': 'Số nhân CPU',
#     'pc': 'Camera sau (MP)',
#     'px_height': 'Chiều cao màn hình (px)',
#     'px_width': 'Chiều rộng màn hình (px)',
#     'ram': 'RAM (MB)',
#     'sc_h': 'Chiều cao màn hình (cm)',
#     'sc_w': 'Chiều rộng màn hình (cm)',
#     'talk_time': 'Thời gian đàm thoại (giờ)',
#     'three_g': 'Hỗ trợ 3G (0/1)',
#     'touch_screen': 'Màn hình cảm ứng (0/1)',
#     'wifi': 'Wifi (0/1)',
# }

# class KMeans:
#     def __init__(self, num_clusters=8, max_iters=300):
#         self.num_clusters = num_clusters
#         self.max_iters = max_iters
#         self.centroids = None
#         self.clusters = None

#     def fit(self, X):
#         self.centroids = X[np.random.choice(range(len(X)), self.num_clusters, replace=False)]
#         for _ in range(self.max_iters):
#             self.clusters = [[] for _ in range(self.num_clusters)]
#             for x in X:
#                 distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
#                 cluster_idx = np.argmin(distances)
#                 self.clusters[cluster_idx].append(x)
#             new_centroids = []
#             for cluster in self.clusters:
#                 new_centroid = np.mean(cluster, axis=0)
#                 new_centroids.append(new_centroid)
            
#             if np.allclose(new_centroids, self.centroids):
#                 break
#             self.centroids = new_centroids

#     def predict(self, X):
#         cluster_labels = [np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids]) for x in X]
#         return cluster_labels

# def evaluate_model():
#     test_labels = kmeans.predict(test_data)
#     silhouette = silhouette_score(test_data, test_labels)
#     davies_bouldin = davies_bouldin_score(test_data, test_labels)
#     evaluation_label.configure(text=f"Độ đo Silhouette: {silhouette:.10f}\nĐộ đo Davies-Bouldin: {davies_bouldin:.10f}")

# num_clusters = 10
# b = []
# for i in range(2, num_clusters + 1):
#     kmeans = KMeans(num_clusters=i)
#     kmeans.fit(train_data)
#     test_labels = kmeans.predict(test_data)
#     silhouette = silhouette_score(test_data, test_labels)
#     davies_bouldin = davies_bouldin_score(test_data, test_labels)
#     new_list = [silhouette, davies_bouldin, i]
#     b.append(new_list)
# max_b = max(b)
# print(max_b)

# kmeans = KMeans(num_clusters=2)
# kmeans.fit(train_data)

# form = Tk()
# form.title("Phân cụm dữ liệu điện thoại di động")
# form.geometry("500x800")  # Tăng kích thước form để hiển thị tốt hơn

# def predict_cluster():
#     new_sample = []
#     for entry in entry_fields:
#         value = entry.get()
#         if value == "":
#             messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin vào các trường.")
#             return
#         try:
#             new_sample.append(float(value))
#         except ValueError:
#             messagebox.showerror("Lỗi", f"Giá trị không hợp lệ: {value}")
#             return

#     cluster_label = kmeans.predict(np.array([new_sample]))[0]
#     result_label.configure(text=f"Kết quả phân cụm: Nhóm {cluster_label + 1}")

# # Tạo frame để chứa các widget với thanh cuộn
# frame = Frame(form)
# canvas = Canvas(frame)
# scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
# scrollable_frame = Frame(canvas)

# scrollable_frame.bind(
#     "<Configure>",
#     lambda e: canvas.configure(
#         scrollregion=canvas.bbox("all")
#     )
# )

# canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
# canvas.configure(yscrollcommand=scrollbar.set)

# entry_fields = []

# # Thêm label giới thiệu
# intro_label = Label(scrollable_frame, text="Nhập thông tin điện thoại", font=("Arial", 12, "bold"))
# intro_label.grid(row=0, column=0, columnspan=2, pady=10)

# # Tạo các trường nhập liệu với nhãn tiếng Việt
# for i, (eng_label, viet_label) in enumerate(vietnamese_labels.items(), start=1):
#     label = Label(scrollable_frame, text=viet_label, anchor='e', justify=LEFT, wraplength=200)
#     label.grid(row=i, column=0, sticky='e', padx=5, pady=2)
    
#     entry = Entry(scrollable_frame)
#     entry.grid(row=i, column=1, padx=5, pady=2)
#     entry_fields.append(entry)

# # Tạo các nút bấm
# predict_button = Button(scrollable_frame, text="Dự đoán phân cụm", command=predict_cluster)
# predict_button.grid(row=len(vietnamese_labels)+1, column=0, columnspan=2, pady=10)

# evaluate_button = Button(scrollable_frame, text="Đánh giá mô hình", command=evaluate_model)
# evaluate_button.grid(row=len(vietnamese_labels)+2, column=0, columnspan=2, pady=5)

# result_label = Label(scrollable_frame, text="", font=("Arial", 10, "bold"))
# result_label.grid(row=len(vietnamese_labels)+3, column=0, columnspan=2)

# evaluation_label = Label(scrollable_frame, text="")
# evaluation_label.grid(row=len(vietnamese_labels)+4, column=0, columnspan=2)

# # Pack the scroll frame
# frame.pack(fill="both", expand=True)
# canvas.pack(side="left", fill="both", expand=True)
# scrollbar.pack(side="right", fill="y")

# form.mainloop()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Đọc và xử lý dữ liệu
df = pd.read_csv('train.csv')
X = df.drop('price_range', axis=1)
y = df['price_range']

# Feature engineering
X = add_engineered_features(X)

# Chọn features tốt nhất
best_features = select_best_features(X, y)
X = X[best_features]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
robust_scaler = RobustScaler()
power_transformer = PowerTransformer()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_robust = robust_scaler.fit_transform(X_train_scaled)
X_test_robust = robust_scaler.transform(X_test_scaled)

X_train_transformed = power_transformer.fit_transform(X_train_robust)
X_test_transformed = power_transformer.transform(X_test_robust)

# Cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

# Tìm tham số tối ưu
param_grid = {
    'n_neighbors': range(1, 31, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_balanced, y_train_balanced)
best_params = grid_search.best_params_

# Tạo ensemble model
estimators = [
    ('knn', KNeighborsClassifier(**best_params)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

final_model = VotingClassifier(estimators=estimators, voting='soft')
final_model.fit(X_train_balanced, y_train_balanced)

# Đánh giá
y_pred = final_model.predict(X_test_transformed)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))