"""
K-NEAREST NEIGHBORS - PHÂN LOẠI CHỮ CÁI (LETTER RECOGNITION)
Dataset: UCI Letter Recognition
- 26 classes: A to Z
- 16 features: statistical moments and edge counts
- 20,000 samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ==================== BƯỚC 1: THU THẬP VÀ CHUẨN BỊ DỮ LIỆU ====================
print("=" * 80)
print("BƯỚC 1: THU THẬP VÀ CHUẨN BỊ DỮ LIỆU")
print("=" * 80)

# Đọc dữ liệu từ file locall
print("\n📥 Đang đọc dữ liệu từ file local...")

# Đặt tên file của bạn tại đây
FILE_PATH = "/Users/tuong/Documents/MONHOCDAIHOC/Khai phá data/ai_practice_prj/fai_practice_prj/p01_KNN/codep01/letter/letter-recognition.data"  # Thay đổi đường dẫn nếu cần

# Tên các cột theo documentation của UCI
column_names = ['letter', 'x-box', 'y-box', 'width', 'height', 'onpix', 
                'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 
                'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']

try:
    # Đọc file CSV/TXT
    df = pd.read_csv(FILE_PATH, names=column_names, header=None)
    X = df.iloc[:, 1:].values  # Tất cả features (cột 2-17)
    y = df.iloc[:, 0].values    # Labels (cột 1: chữ cái)
    print(f"✓ Đọc dữ liệu thành công từ file: {FILE_PATH}")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file: {FILE_PATH}")
    print("📌 Vui lòng:")
    print("   1. Đảm bảo file 'etter-recognition.data' nằm cùng thư mục với code")
    print("   2. Hoặc thay đổi FILE_PATH thành đường dẫn đầy đủ đến file")
    print("   Ví dụ: FILE_PATH = 'C:/Users/YourName/Desktop/letter-recognition.data'")
    exit()
except Exception as e:
    print(f"❌ Lỗi khi đọc file: {e}")
    exit()

# Tạo DataFrame để xem
feature_names = ['x-box', 'y-box', 'width', 'height', 'onpix', 
                 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 
                 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']

df = pd.DataFrame(X, columns=feature_names)
df['letter'] = y

print(f"\n📊 Thông tin dữ liệu:")
print(f"- Số mẫu: {len(df)}")
print(f"- Số đặc trưng: {X.shape[1]}")
print(f"- Số classes (A-Z): {len(np.unique(y))}")
print(f"- Classes: {sorted(np.unique(y))}")

print(f"\n📋 5 mẫu đầu tiên:")
print(df.head())

print(f"\n📈 Thống kê mô tả:")
print(df.describe())

print(f"\n🔍 Kiểm tra dữ liệu thiếu:")
print(f"Số giá trị null: {df.isnull().sum().sum()}")

print(f"\n📊 Phân bố các chữ cái (5 chữ đầu):")
print(df['letter'].value_counts().head())


# ==================== BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ====================
print("\n" + "=" * 80)
print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 80)

# 2.1: Kiểm tra và xử lý dòng trùng TRƯỚC KHI chia dữ liệu
print("\n[2.1] Kiểm tra dòng trùng:")
duplicates_count = df.duplicated().sum()
print(f"Số dòng trùng: {duplicates_count}")

if duplicates_count > 0:
    print(f"✓ Tìm thấy {duplicates_count} dòng trùng")
    df = df.drop_duplicates()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(f"✓ Sau khi xóa trùng: {len(df)} mẫu")
else:
    print("✓ Không có dòng trùng trong dataset")

# 2.2: Encode labels (chuyển chữ cái thành số)
print("\n[2.2] Encode labels (A-Z → 0-25):")
print(f"Labels trước khi encode: {sorted(np.unique(y))[:5]}... (hiển thị 5 chữ đầu)")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Labels sau khi encode: {np.unique(y_encoded)[:5]}... (0-25)")
print(f"✓ Mapping ví dụ: A={label_encoder.transform(['A'])[0]}, B={label_encoder.transform(['B'])[0]}, Z={label_encoder.transform(['Z'])[0]}")

# Cập nhật y
y = y_encoded

print(f"✓ Shape của X: {X.shape}")
print(f"✓ Shape của y: {y.shape}")

# 2.3: Chia dữ liệu train/test (70/30)
print("\n[2.3] Chia dữ liệu train/test (70/30):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✓ Tập train: {len(X_train)} mẫu ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Tập test: {len(X_test)} mẫu ({len(X_test)/len(X)*100:.1f}%)")

# 2.4: Chuẩn hóa CHỈ features (X)
print("\n[2.4] Chuẩn hóa features (StandardScaler):")
print("⚠️  CHỈ chuẩn hóa features (X), KHÔNG chuẩn hóa labels (y)!")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Mean của X_train sau chuẩn hóa: {X_train_scaled.mean(axis=0)[:3].round(2)}... (gần 0)")
print(f"✓ Std của X_train sau chuẩn hóa: {X_train_scaled.std(axis=0)[:3].round(2)}... (gần 1)")


# ==================== BƯỚC 3: CÀI ĐẶT THUẬT TOÁN KNN ====================
print("\n" + "=" * 80)
print("BƯỚC 3: CÀI ĐẶT THUẬT TOÁN K-NEAREST NEIGHBORS")
print("=" * 80)

class KNearestNeighbors:
    """Cài đặt thuật toán K-Nearest Neighbors từ đầu"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """Lưu dữ liệu training"""
        self.X_train = X_train
        self.y_train = y_train
        print(f"✓ Đã lưu {len(X_train)} mẫu training")
    
    def euclidean_distance(self, x1, x2):
        """Tính khoảng cách Euclidean"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict_single(self, x):
        """Dự đoán cho 1 điểm dữ liệu"""
        # Tính khoảng cách đến tất cả điểm training
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Lấy k điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Voting: chọn label xuất hiện nhiều nhất
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """Dự đoán cho nhiều điểm dữ liệu"""
        predictions = []
        total = len(X)
        for i, x in enumerate(X):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  Đang dự đoán: {i+1}/{total} mẫu ({(i+1)/total*100:.1f}%)")
            predictions.append(self.predict_single(x))
        return np.array(predictions)

# Khởi tạo với k=5 (tốt hơn cho 26 classes)
knn = KNearestNeighbors(k=5)
print(f"\n✓ Khởi tạo mô hình KNN với k={knn.k}")
print(f"  (Với 26 classes, k=5 thường cho kết quả tốt hơn k=3)")


# ==================== BƯỚC 4: HUẤN LUYỆN VÀ KIỂM THỬ ====================
print("\n" + "=" * 80)
print("BƯỚC 4: HUẤN LUYỆN VÀ KIỂM THỬ")
print("=" * 80)

# Huấn luyện
print("\n🎯 Đang huấn luyện mô hình...")
knn.fit(X_train_scaled, y_train)
print("✓ Hoàn thành huấn luyện!")

# Dự đoán (có thể mất vài phút với 20000 mẫu)
print(f"\n🔮 Đang dự đoán trên {len(X_test_scaled)} mẫu test...")
print("   (Quá trình này có thể mất vài phút...)")
y_pred = knn.predict(X_test_scaled)
print("✓ Hoàn thành dự đoán!")

# Đánh giá
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 ĐỘ CHÍNH XÁC (ACCURACY): {accuracy * 100:.2f}%")

# Confusion Matrix (chỉ hiển thị 10x10 đầu tiên do quá lớn)
cm = confusion_matrix(y_test, y_pred)
print(f"\n📈 Kích thước Ma trận nhầm lẫn: {cm.shape}")

# Metrics tổng quát
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n📊 METRICS TỔNG QUÁT (Macro Average):")
print(f"  - Precision: {precision_macro*100:.2f}%")
print(f"  - Recall: {recall_macro*100:.2f}%")
print(f"  - F1-Score: {f1_macro*100:.2f}%")

# Chi tiết một số chữ cái
print(f"\n📋 CHI TIẾT MỘT SỐ CHỮ CÁI:")
letters = label_encoder.classes_
for i in [0, 1, 2, 25]:  # A, B, C, Z
    indices = np.where(y_test == i)[0]
    if len(indices) > 0:
        y_test_letter = y_test[indices]
        y_pred_letter = y_pred[indices]
        acc = accuracy_score(y_test_letter, y_pred_letter)
        print(f"  Letter {letters[i]}: Accuracy = {acc*100:.2f}%")

# So sánh một số dự đoán
print(f"\n🔍 SO SÁNH 10 DỰ ĐOÁN ĐẦU TIÊN:")
print(f"{'STT':<5} {'Thực tế':<10} {'Dự đoán':<10} {'Kết quả':<10}")
print("-" * 40)
for i in range(min(10, len(y_test))):
    actual = letters[y_test[i]]
    predicted = letters[y_pred[i]]
    result = "✓ Đúng" if y_test[i] == y_pred[i] else "✗ Sai"
    print(f"{i+1:<5} {actual:<10} {predicted:<10} {result:<10}")


# ==================== BƯỚC 5: TỐI ƯU HÓA ====================
print("\n" + "=" * 80)
print("BƯỚC 5: TỐI ƯU HÓA - TÌM GIÁ TRỊ K TỐI ƯU")
print("=" * 80)

print("\n⚠️  LƯU Ý: Với dataset lớn (20000 mẫu), việc thử nhiều k tốn thời gian.")
print("   Chúng ta sẽ thử k từ 1 đến 10 trên tập test nhỏ hơn.\n")

# Lấy subset nhỏ để test nhanh
n_subset = min(1000, len(X_test_scaled))
X_test_subset = X_test_scaled[:n_subset]
y_test_subset = y_test[:n_subset]

k_values = range(1, 11)
accuracies = []

print(f"Đang thử nghiệm k từ 1 đến 10 trên {n_subset} mẫu test...")
for k in k_values:
    knn_temp = KNearestNeighbors(k=k)
    knn_temp.fit(X_train_scaled, y_train)
    print(f"\nk={k}:")
    y_pred_temp = knn_temp.predict(X_test_subset)
    acc = accuracy_score(y_test_subset, y_pred_temp)
    accuracies.append(acc)
    print(f"  Accuracy = {acc*100:.2f}%")

best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\n🏆 GIÁ TRỊ K TỐI ƯU: k={best_k} với accuracy={best_accuracy*100:.2f}%")


# ==================== BƯỚC 6: TRỰC QUAN HÓA ====================
print("\n" + "=" * 80)
print("BƯỚC 6: TRỰC QUAN HÓA KẾT QUẢ")
print("=" * 80)

fig = plt.figure(figsize=(18, 10))

# 1. Phân bố chữ cái trong dataset
ax1 = plt.subplot(2, 3, 1)
letter_counts = pd.Series(y).value_counts().sort_index()
plt.bar(range(26), letter_counts.values, color='steelblue', alpha=0.7)
plt.xlabel('Letter Index (A=0, Z=25)', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Phân bố 26 chữ cái trong dataset', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 2. Confusion Matrix (10x10 đầu tiên)
ax2 = plt.subplot(2, 3, 2)
cm_subset = cm[:10, :10]
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (A-J)', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)

# 3. Accuracy vs K
ax3 = plt.subplot(2, 3, 3)
plt.plot(k_values, [acc*100 for acc in accuracies], marker='o', linewidth=2, markersize=8, color='steelblue')
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
plt.xlabel('Giá trị K', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Accuracy theo giá trị K', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 4. Feature Distribution - Width
ax4 = plt.subplot(2, 3, 4)
plt.hist(X[:, 2], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Width', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Phân bố feature: Width', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 5. Feature Distribution - Height
ax5 = plt.subplot(2, 3, 5)
plt.hist(X[:, 3], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.xlabel('Height', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Phân bố feature: Height', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 6. Accuracy by Letter (10 chữ đầu)
ax6 = plt.subplot(2, 3, 6)
letter_accs = []
for i in range(10):
    indices = np.where(y_test == i)[0]
    if len(indices) > 0:
        acc = accuracy_score(y_test[indices], y_pred[indices])
        letter_accs.append(acc * 100)
    else:
        letter_accs.append(0)

plt.bar(range(10), letter_accs, color='seagreen', alpha=0.7)
plt.xlabel('Letter (A-J)', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Accuracy theo từng chữ cái (A-J)', fontsize=12, fontweight='bold')
plt.xticks(range(10), letters[:10])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('letter_knn_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Đã lưu biểu đồ vào file: letter_knn_results.png")
plt.show()

print("\n" + "=" * 80)
print("✅ HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")
print("=" * 80)
print("\n📌 TÓM TẮT:")
print(f"  - Dataset: 20,000 mẫu, 26 classes (A-Z), 16 features")
print(f"  - Accuracy tổng quát: {accuracy*100:.2f}%")
print(f"  - Giá trị k tốt nhất: {best_k}")
print(f"  - Precision (macro): {precision_macro*100:.2f}%")
print(f"  - Recall (macro): {recall_macro*100:.2f}%")
print(f"  - F1-Score (macro): {f1_macro*100:.2f}%")