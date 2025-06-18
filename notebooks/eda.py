# 1. Extract zilfile to folder BDC ONE PCS
# Path ke file ZIP
zip_path = 'BDC ONE PCS.zip'  # Ganti dengan path file zip kamu
extract_to = 'BDC ONE PCS'  # Ganti dengan direktori tujuan ekstraksi
# Membuat folder tujuan jika belum ada
os.makedirs(extract_to, exist_ok=True)
# Ekstrak file zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("✅ File berhasil diekstrak ke:", extract_to)

# 2. Check folder structures and labels to ensure format consistency and label accuracy
# Based dataset
base_path = 'BDC ONE PCS/Train Data/Train Data'
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

# 3. Display sample picture for noise picture detection and validation
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

# 4. Check for label imbalance based on data amount, to find out the need for data augmentation to balance the ratio
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

# 5. Calculate labels correlation to test the similarity of data
# Pisahkan label jadi list (misal: 'fire+smoke' → ['fire', 'smoke'])
df['label_list'] = df['label'].str.split('+')
# Inisialisasi dan fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
multi_label = mlb.fit_transform(df['label_list'])
# Gabung hasilnya ke dataframe
df_mlb = pd.DataFrame(multi_label, columns=mlb.classes_)
df = pd.concat([df, df_mlb], axis=1)
# Correlation Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(df[mlb.classes_].corr(), annot=True, cmap="Reds", fmt=".2f")
plt.title("Korelasi antar Label")
plt.show()
