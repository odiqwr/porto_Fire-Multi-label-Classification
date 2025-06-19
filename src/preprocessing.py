# Image Preprocessing

# 1. Determining the minority class
label_ratio = df['label'].value_counts(normalize=True)
minority_labels = label_ratio[label_ratio < 0.40].index.tolist()
print("Kelas minoritas terdeteksi:", minority_labels)

# 2. Initial data count calculation
df = df.dropna(subset=['label']).reset_index(drop=True)
# Hitung ulang label
label_counts = dict(Counter(df['label']))
max_count = max(label_counts.values())
print("Jumlah data sebelum oversampling:", label_counts)

# 3. Data Augmentation to handle data imbalanced
# Augmentasi yang digunakan
augment_minor_class = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(p=0.3),
])
# Untuk menyimpan hasil augmentasi
augmented_images = []
augmented_labels = []
# Proses augmentasi per kelas minoritas
for label in minority_labels:
    class_df = df[df['label'] == label]
    current_count = len(class_df)
    gap = max_count - current_count
    print(f"Augmentasi untuk kelas '{label}' sebanyak {gap} gambar")
    for i in range(gap):
        row = class_df.sample(n=1).iloc[0]
        img_path = os.path.join(base_path, row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            # Lakukan augmentasi
            augmented = augment_minor_class(image=img_np)
            aug_img = augmented['image']
            # Nama file baru
            folder_name, file_name = os.path.split(row['filename'])
            new_filename = f"aug_{i}_{file_name}"
            new_rel_path = os.path.join(folder_name, new_filename)
            new_abs_path = os.path.join(base_path, new_rel_path)
            # Simpan gambar baru
            cv2.imwrite(new_abs_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            augmented_images.append(new_rel_path)
            augmented_labels.append(label)
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")
# Update augmentation results on initial dataframe
df_aug = pd.DataFrame({'filename': augmented_images, 'label': augmented_labels})
df = pd.concat([df.reset_index(drop=True), df_aug.reset_index(drop=True)], ignore_index=True)

# 4. Final data count calculation
print("\nDistribusi label setelah augmentasi:")
print(df['label'].value_counts())

# 5. Data visualization after augmentation process with resizing to support uniformity
def show_samples(df, label_name, num=4):
    samples = df[df['label'] == label_name].sample(min(num, len(df[df['label'] == label_name])))
    fig, axs = plt.subplots(1, len(samples), figsize=(15, 5))
    fig.suptitle(f"Contoh Gambar: {label_name}", fontsize=16)
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(base_path, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (180, 180))  # gunakan saat load gambar
        img_normalized = img_resized / 255.0
        axs[i].imshow(img_resized)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
# Tampilkan contoh semua label
for label in df['label'].unique():
    show_samples(df, label)

# 6. Prepare data for modeling with multilab binaryzer
# Pastikan semua nilai label berupa string sebelum split
df['label_list'] = df['label'].astype(str).apply(lambda x: x.split(','))
# Inisialisasi dan fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
label_binarized = mlb.fit_transform(df['label_list'])
# Buat DataFrame hasil binarisasi dan gabungkan ke df
df_mlb = pd.DataFrame(label_binarized, columns=mlb.classes_)
# Gabungkan langsung ke df_asli
df = pd.concat([df, df_mlb], axis=1)
# Hapus kolom duplikat jika ada
df = df.loc[:, ~df.columns.duplicated()]
