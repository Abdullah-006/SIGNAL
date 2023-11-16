import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

# Load your sign language dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode the string labels to numerical labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.3, shuffle=True,
                                                    stratify=labels_encoded)

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
model.add(Dense(3, activation='softmax'))  # num_classes is the number of sign language classes

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Track accuracy during training
train_accuracy = []
val_accuracy = []

# Train the model
for epoch in range(10):  # Modify the number of epochs as needed
    history = model.fit(flatten_train, y_train, epochs=10, batch_size=32, validation_data=(flatten_test, y_test))

    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]

    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)

# Evaluate the model on the testing set
y_pred_prob = model.predict(flatten_test)
y_pred = np.argmax(y_pred_prob, axis=1)
score = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(score * 100))

# Save the model
model.save('model.p')

# Plot the accuracy graph
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_graph.jpg', format='jpg')
plt.show()
