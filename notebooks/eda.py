# 1. Extract zilfile to folder BDC ONE PCS
# Path ke file ZIP
zip_path = 'big-data-competition-statistics-explore-2024.zip'  # Ganti dengan path file zip kamu
extract_to = 'PCS'  # Ganti dengan direktori tujuan ekstraksi
# Membuat folder tujuan jika belum ada
os.makedirs(extract_to, exist_ok=True)
# Ekstrak file zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("✅ File berhasil diekstrak ke:", extract_to)

# 2. Check folder structures and labels to ensure format consistency and label accuracy
# Path utama dataset
base_path = 'PCS/Train Data/Train Data'
# Ambil semua folder label
label_folders = os.listdir(base_path)
label_folders = [folder for folder in label_folders if os.path.isdir(os.path.join(base_path, folder))]
# Baca file gambar dan buat dataframe
data = []
for label in label_folders:
    folder_path = os.path.join(base_path, label)
    for file in os.listdir(folder_path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            data.append({
                'filename': os.path.join(label, file),  # Simpan path relatif
                'label': label
            })
df = pd.DataFrame(data)
df.info

# 3. Check unreadable image
# Simpan path yang gagal dibaca
invalid_paths = []
sizes = []
for idx, row in df.iterrows():
    img_path = os.path.join(base_path, row['filename'])
    try:
        with Image.open(img_path) as img:
            sizes.append(img.size)  # (width, height)
    except Exception as e:
        print(f"Gagal membaca: {img_path} | Error: {e}")
        invalid_paths.append(row['filename'])  # Simpan filename yang gagal
        try:
            os.remove(img_path)
            print(f"File dihapus: {img_path}")
        except Exception as delete_error:
            print(f"Gagal menghapus {img_path} | Error: {delete_error}")
# Perbarui dataframe: hapus baris dengan filename yang error
initial_len = len(df)
df = df[~df['filename'].isin(invalid_paths)].reset_index(drop=True)
print(f"Jumlah baris dihapus dari DataFrame: {initial_len - len(df)}")

# 4. Display sample picture for noise picture detection and validation
def show_samples(df, label_name, num=4):
    samples = df[df['label'] == label_name].sample(min(num, len(df[df['label'] == label_name])))
    fig, axs = plt.subplots(1, len(samples), figsize=(15, 5))
    fig.suptitle(f"Contoh Gambar: {label_name}", fontsize=16)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(base_path, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
# Tampilkan contoh semua label
for label in df['label'].unique():
    show_samples(df, label)

# 5. Calculate image resolution to improve uniformity
size_counts = Counter(sizes)
print("Ukuran paling umum:")
print(size_counts.most_common(5))

# 6. Visualization of the Number of Label Gap Data
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette='viridis')
plt.title("Distribusi Gambar Berdasarkan Label Folder")
plt.xlabel("Label")
plt.ylabel("Jumlah")
plt.xticks(rotation=15)
plt.show()

# 7. Check for label imbalance based on data amount, to find out the need for data augmentation to balance the ratio
# Hitung rasio label
label_ratio = df['label'].value_counts(normalize=True)
# Pie plot
plt.figure(figsize=(6, 6))
plt.pie(label_ratio.values, labels=label_ratio.index, autopct='%1.1f%%', colors=sns.color_palette('flare', n_colors=len(label_ratio)))
plt.title("Distribusi Label (Pie Plot)")
plt.axis('equal')  # Agar lingkaran proporsional
plt.show()
# Cetak nilai rasio
print("Rasio label terhadap total:")
print(label_ratio)

# 8. Calculate labels correlation to test the similarity of data
# Pisahkan label jadi list (misal: 'fire+smoke' → ['fire', 'smoke'])
df['label_list'] = df['label'].str.split('+')
# Inisialisasi dan fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
multi_label = mlb.fit_transform(df['label_list'])
# Gabung hasilnya ke dataframe
df_mlb = pd.DataFrame(multi_label, columns=mlb.classes_)
# Correlation Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(df_mlb[mlb.classes_].corr(), annot=True, cmap="Reds", fmt=".2f")
plt.title("Korelasi antar Label")
plt.show()
