# 1. Splitting data for training and testing
# Misal kolom label binarisasi: 'Fire', 'Smoke', 'None', 'Smoke and Fire'
label_columns = ['Fire', 'Smoke', 'None', 'Smoke and Fire']
# define object
X = df['filename']  # path gambar
y = df[label_columns]  # multi-label target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.values)
# show data labels for training
print (y_train)

# 2. Generate image data for training and testing
IMG_SIZE = (180, 180)
BATCH_SIZE = 24
# Buat ImageDataGenerator (tanpa augmentasi tambahan di sini)
datagen = ImageDataGenerator(rescale=1./255)
# generate data train
train_gen = datagen.flow_from_dataframe(
    dataframe=pd.concat([X_train, y_train], axis=1),
    x_col='filename',
    y_col=label_columns,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',  # karena multi-label
    shuffle=True
)
# generate data test
val_gen = datagen.flow_from_dataframe(
    dataframe=pd.concat([X_test, y_test], axis=1),
    x_col='filename',
    y_col=label_columns,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# 3. Build Deep Learning Model Architecture of Convolution Neural Network
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(180,180,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(16, (3,3), activation='relu'),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(8, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_columns), activation='sigmoid')  # sigmoid untuk multi-label
])
# model compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# model summary
model.summary()

# 4. Training deep learning model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)
