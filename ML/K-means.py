from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Đọc dữ liệu
df = pd.read_csv('train.csv')

# Danh sách features và mô tả
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

# Chuẩn bị dữ liệu
X = df.drop('price_range', axis=1)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Khởi tạo và huấn luyện mô hình K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Tạo giao diện
root = Tk()
root.title("Phân cụm điện thoại")

# Tạo frame cuộn
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = Frame(canvas)
canvas.create_window((0,0), window=second_frame, anchor="nw")

# Style
style = ttk.Style()
style.configure("TLabel", padding=5, font=('Helvetica', 10))
style.configure("TEntry", padding=5)
style.configure("TButton", padding=5, font=('Helvetica', 10))

# Dictionary để lưu các entry fields
entries = {}

def create_tooltip(widget, text):
    def show_tooltip(event):
        tooltip = Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = Label(tooltip, text=text, justify=LEFT,
                     background="#ffffe0", relief=SOLID, borderwidth=1)
        label.pack()
        
        def hide_tooltip():
            tooltip.destroy()
        
        widget.bind('<Leave>', lambda e: hide_tooltip())
        tooltip.bind('<Leave>', lambda e: hide_tooltip())
        
    widget.bind('<Enter>', show_tooltip)

# Tạo các trường nhập liệu
row = 0
for feature, description in features_info.items():
    frame = Frame(second_frame)
    frame.grid(row=row, column=0, sticky='w', padx=5, pady=2)
    
    label = ttk.Label(frame, text=f"{description}:")
    label.pack(side=LEFT)
    
    entry = ttk.Entry(frame)
    entry.pack(side=LEFT, padx=5)
    entries[feature] = entry
    
    # Thêm tooltip cho các trường binary
    if feature in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
        create_tooltip(entry, "Nhập 0 hoặc 1")
    
    row += 1

def validate_inputs():
    for feature, entry in entries.items():
        value = entry.get().strip()
        if not value:
            messagebox.showerror("Lỗi", f"Vui lòng nhập giá trị cho {features_info[feature]}")
            return False
        try:
            float(value)
        except ValueError:
            messagebox.showerror("Lỗi", f"Giá trị không hợp lệ cho {features_info[feature]}")
            return False
    return True

def predict_cluster():
    if not validate_inputs():
        return
    
    # Lấy giá trị từ các entry
    input_data = []
    for feature in features_info.keys():
        input_data.append(float(entries[feature].get()))
    
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform([input_data])
    
    # Dự đoán cụm
    cluster = kmeans.predict(input_scaled)[0]
    
    # Phân tích đặc điểm của cụm
    cluster_center = kmeans.cluster_centers_[cluster]
    
    # Xác định mức giá dựa trên đặc điểm của cụm
    price_ranges = {
        0: "Thấp (< 250$)",
        1: "Trung bình (250$ - 500$)",
        2: "Cao (500$ - 750$)",
        3: "Rất cao (> 750$)"
    }
    
    result_text = f"Điện thoại thuộc cụm {cluster + 1}\n"
    result_text += f"Phân khúc giá dự đoán: {price_ranges[cluster]}"
    messagebox.showinfo("Kết quả phân cụm", result_text)

def clear_entries():
    for entry in entries.values():
        entry.delete(0, END)

# Tạo frame cho các nút
button_frame = Frame(second_frame)
button_frame.grid(row=row, column=0, pady=10)

# Thêm các nút
predict_button = ttk.Button(button_frame, text="Phân cụm", command=predict_cluster)
predict_button.pack(side=LEFT, padx=5)

clear_button = ttk.Button(button_frame, text="Xóa", command=clear_entries)
clear_button.pack(side=LEFT, padx=5)

# Cấu hình kích thước cửa sổ
root.geometry("500x600")

root.mainloop()

# class CustomKMeans:
#     def __init__(self, n_clusters, max_iters=100):
#         self.n_clusters = n_clusters
#         self.max_iters = max_iters
#         self.centroids = None
        
#     def fit(self, X):
#         # Khởi tạo centroids ngẫu nhiên
#         random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
#         self.centroids = X[random_indices]
        
#         for _ in range(self.max_iters):
#             # Gán điểm vào các cụm
#             clusters = self._assign_clusters(X)
            
#             # Lưu centroids cũ để kiểm tra hội tụ
#             old_centroids = self.centroids.copy()
            
#             # Cập nhật centroids
#             for i in range(self.n_clusters):
#                 points_in_cluster = X[clusters == i]
#                 if len(points_in_cluster) > 0:
#                     self.centroids[i] = points_in_cluster.mean(axis=0)
            
#             # Kiểm tra hội tụ
#             if np.all(old_centroids == self.centroids):
#                 break
                
#     def _assign_clusters(self, X):
#         # Tính khoảng cách từ mỗi điểm đến các centroids
#         distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
#         # Gán điểm vào cụm gần nhất
#         return np.argmin(distances, axis=0)
    
#     def predict(self, X):
#         return self._assign_clusters(X)