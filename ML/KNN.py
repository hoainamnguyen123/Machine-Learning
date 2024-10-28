import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# Đọc dữ liệu
df = pd.read_csv('train.csv')
X = df.drop('price_range', axis=1)
y = df['price_range']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tìm K tối ưu
k_values = range(1, 31, 2)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

best_k = k_values[accuracies.index(max(accuracies))]
print(f"K tối ưu: {best_k}")

# Vẽ biểu đồ accuracy theo K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'bo-')
plt.xlabel('Giá trị K')
plt.ylabel('Độ chính xác')
plt.title('Độ chính xác theo giá trị K')
plt.grid(True)
plt.show()

# Chọn features quan trọng
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()].tolist()
print("Features quan trọng nhất:", selected_features)

# Huấn luyện mô hình cuối cùng với K tối ưu
final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
final_knn.fit(X_train_scaled, y_train)

# Đánh giá mô hình
y_pred_final = final_knn.predict(X_test_scaled)
print("\nKết quả đánh giá mô hình:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred_final)*100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred_final))

# Thông tin các trường
features_info = {
    'battery_power': 'Dung lượng pin (mAh)',
    'blue': 'Bluetooth (0: Không, 1: Có)',
    'clock_speed': 'Tốc độ xử lý (GHz)',
    'dual_sim': 'Hỗ trợ 2 SIM (0: Không, 1: Có)',
    'fc': 'Camera trước (MP)',
    'four_g': 'Hỗ trợ 4G (0: Không, 1: Có)',
    'int_memory': 'Bộ nhớ trong (GB)',
    'm_dep': 'Độ dày máy (cm)',
    'mobile_wt': 'Trọng lượng (g)',
    'n_cores': 'Số nhân CPU',
    'pc': 'Camera chính (MP)',
    'px_height': 'Chiều cao màn hình (px)',
    'px_width': 'Chiều rộng màn hình (px)',
    'ram': 'RAM (MB)',
    'sc_h': 'Chiều cao màn hình (cm)',
    'sc_w': 'Chiều rộng màn hình (cm)',
    'talk_time': 'Thời gian đàm thoại (giờ)',
    'three_g': 'Hỗ trợ 3G (0: Không, 1: Có)',
    'touch_screen': 'Màn hình cảm ứng (0: Không, 1: Có)',
    'wifi': 'Wifi (0: Không, 1: Có)'
}

# Hàm dự đoán
def predict_price_range(features):
    features_scaled = scaler.transform([features])
    prediction = final_knn.predict(features_scaled)
    proba = final_knn.predict_proba(features_scaled)
    return prediction[0], proba[0]

# Giao diện
root = Tk()
root.title("Dự đoán giá điện thoại bằng KNN")

# Tạo frame chính với thanh cuộn
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

canvas = Canvas(main_frame)
scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Tạo các trường nhập liệu
entries = {}
for feature, label_text in features_info.items():
    frame = Frame(scrollable_frame)
    frame.pack(anchor='w', padx=5, pady=2)
    
    label = Label(frame, text=f"{label_text}:")
    label.pack(side=LEFT)
    
    entry = Entry(frame)
    entry.pack(side=LEFT, padx=5)
    entries[feature] = entry

def handle_predict():
    try:
        features = [float(entries[feature].get()) for feature in features_info.keys()]
        prediction, probabilities = predict_price_range(features)
        
        result_text = f"Dự đoán khoảng giá: {prediction}\n\n"
        result_text += "Xác suất cho từng khoảng giá:\n"
        for i, prob in enumerate(probabilities):
            result_text += f"Khoảng giá {i}: {prob*100:.2f}%\n"
            
        messagebox.showinfo("Kết quả Dự đoán", result_text)
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập số hợp lệ cho tất cả các trường")

# Thêm nút dự đoán
predict_button = Button(scrollable_frame, text="Dự đoán giá", command=handle_predict)
predict_button.pack(pady=10)

# Pack thanh cuộn và canvas
canvas.pack(side=LEFT, fill=BOTH, expand=1)
scrollbar.pack(side=RIGHT, fill=Y)

# Cấu hình cửa sổ
root.geometry("500x750")
root.mainloop()

# class MyKNN:
#     def __init__(self, k=3):
#         self.k = k
        
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
        
#     def euclidean_distance(self, x1, x2):
#         return np.sqrt(np.sum((x1 - x2) ** 2))
    
#     def predict(self, X):
#         predictions = []
        
#         for x in X:
#             # Tính khoảng cách từ điểm cần dự đoán đến tất cả điểm train
#             distances = []
#             for x_train in self.X_train:
#                 distance = self.euclidean_distance(x, x_train)
#                 distances.append(distance)
            
#             # Lấy k láng giềng gần nhất
#             k_indices = np.argsort(distances)[:self.k]
#             k_nearest_labels = [self.y_train[i] for i in k_indices]
            
#             # Dự đoán nhãn bằng voting
#             most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
#             predictions.append(most_common)
            
#         return np.array(predictions)

