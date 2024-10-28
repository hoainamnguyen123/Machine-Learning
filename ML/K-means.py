from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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


def analyze_price_segments(df, kmeans):
    # Thêm cột cluster vào dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = kmeans.labels_
    
    # Tính giá trị trung bình của các features quan trọng cho mỗi cụm 
    cluster_means = df_with_clusters.groupby('cluster').mean()
    
    # Các features quan trọng ảnh hưởng đến giá
    price_features = ['ram', 'battery_power', 'int_memory', 'pc', 'px_width', 'px_height']
    
    # Tính điểm tổng hợp cho mỗi cụm dựa trên các features quan trọng
    cluster_scores = {}
    for cluster in range(kmeans.n_clusters):
        score = sum(cluster_means.loc[cluster, feature] for feature in price_features)
        cluster_scores[cluster] = score
    
    # Sắp xếp các cụm theo điểm từ thấp đến cao
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])
    
    # Gán phân khúc giá cho các cụm
    price_segments = {}
    price_ranges = ["Thấp (< 250$)", "Trung bình (250$ - 500$)", 
                   "Cao (500$ - 750$)", "Rất cao (> 750$)"]
    
    for i, (cluster, _) in enumerate(sorted_clusters):
        price_segments[cluster] = price_ranges[i]
    
    return price_segments, cluster_means

price_segments, cluster_characteristics = analyze_price_segments(df, kmeans)

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
    
    input_data = []
    for feature in features_info.keys():
        input_data.append(float(entries[feature].get()))
    
    input_scaled = scaler.transform([input_data])
    cluster = kmeans.predict(input_scaled)[0]
    
    # Phân tích đặc điểm của cụm
    result_text = "";
    
    # Thêm một số đặc điểm nổi bật của cụm
    result_text += "Đặc điểm nổi bật của cụm:\n"
    important_features = ['ram', 'battery_power', 'int_memory', 'pc','px_width','px_height']
    for feature in important_features:
        value = cluster_characteristics.loc[cluster, feature]
        result_text += f"- {features_info[feature]}: {value:.2f}\n"
    
    # Thêm phân khúc giá
    result_text += f"\nPhân khúc giá dự đoán: {price_segments[cluster]}"
    
    messagebox.showinfo("Kết quả phân cụm", result_text)



def clear_entries():
    for entry in entries.values():
        entry.delete(0, END)


# Tính độ đo Silhouette và Davies-Bouldin
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
davies_bouldin_avg = davies_bouldin_score(X_scaled, kmeans.labels_)

print(f"Độ đo Silhouette: {silhouette_avg:.9f}")
print(f"Độ đo Davies-Bouldin: {davies_bouldin_avg:.9f}")
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