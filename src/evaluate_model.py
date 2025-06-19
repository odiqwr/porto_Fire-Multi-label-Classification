# 1. Plot Accuracy and Loss to detecting loss reduction consistency
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")
plt.show()

# 2. Classification report 
# Prediksi
y_pred = model.predict(val_gen)
y_pred_binary = (y_pred > 0.5).astype(int)
# Evaluasi
print(classification_report(y_test, y_pred_binary, target_names=label_columns))
# Save Model
model.save("cnn_fire_smoke_model.h5")

# 3. Confusion matrix to calculate the number of true and false differences between the actual and predicted results.
# calculate yes and no matrix for every label
mcm = multilabel_confusion_matrix(y_test, y_pred_binary)
# define label names
label_names = ['Fire', 'Smoke', 'None', 'Smoke and Fire']
# confusion matrix visualizaton with looping based on data labels
for i, label in enumerate(label_names):
    tn, fp, fn, tp = mcm[i].ravel()
    cm_array = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ['No', 'Yes'])
    plt.yticks([0.5, 1.5], ['No', 'Yes'])
    plt.show()
