import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load your sign language dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Select 1000 images from each class
num_samples_per_class = 1000
selected_data = []
selected_labels = []
unique_labels = np.unique(labels)
for label in unique_labels:
    class_data = data[labels == label][:num_samples_per_class]
    class_labels = labels[labels == label][:num_samples_per_class]
    selected_data.append(class_data)
    selected_labels.append(class_labels)

selected_data = np.concatenate(selected_data)
selected_labels = np.concatenate(selected_labels)

# Encode the string labels to numerical labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(selected_labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(selected_data, labels_encoded, test_size=0.3, shuffle=True, stratify=labels_encoded)

# Resize and preprocess the input images for the VGG16 model
x_train_resized = np.asarray([np.array(Image.fromarray(img).resize((224, 224))) for img in x_train])
x_test_resized = np.asarray([np.array(Image.fromarray(img).resize((224, 224))) for img in x_test])

# Convert grayscale images to RGB format
if x_train_resized.ndim == 3:
    x_train_resized = np.stack((x_train_resized,) * 3, axis=-1)
if x_test_resized.ndim == 3:
    x_test_resized = np.stack((x_test_resized,) * 3, axis=-1)

x_train_preprocessed = preprocess_input(x_train_resized)
x_test_preprocessed = preprocess_input(x_test_resized)

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from the images using the VGG16 model
features_train = base_model.predict(x_train_preprocessed)
features_test = base_model.predict(x_test_preprocessed)

# Flatten the extracted features
flatten_train = features_train.reshape(features_train.shape[0], -1)
flatten_test = features_test.reshape(features_test.shape[0], -1)

# Build a classifier model on top of the pre-trained features
model = Sequential()
model.add(Flatten(input_shape=flatten_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(len(unique_labels), activation='softmax'))  # num_classes is the number of sign language classes

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(flatten_train, y_train, epochs=10, batch_size=32, validation_data=(flatten_test, y_test))

# Evaluate the model on the testing set
y_pred_prob = model.predict(flatten_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('F1 Score: {:.2f}'.format(f1))
print('Confusion Matrix:')
print(confusion_mat)

# Display confusion matrix
class_labels = [label_encoder.classes_[i] for i in range(len(label_encoder.classes_))]
fig, ax = plt.subplots()
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
            yticklabels=class_labels, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
plt.show()

# Plot the accuracy graph
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Calculate train-test split percentage
train_percentage = len(x_train) / len(selected_data) * 100
test_percentage = len(x_test) / len(selected_data) * 100
print('Train percentage: {:.2f}%'.format(train_percentage))
print('Test percentage: {:.2f}%'.format(test_percentage))
