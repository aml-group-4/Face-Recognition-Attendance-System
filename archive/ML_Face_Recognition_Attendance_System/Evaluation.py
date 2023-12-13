import tensorflow as tf
import numpy as np
import cv2
import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
from keras_facenet import FaceNet
from keras.applications import resnet
from keras import layers, Model
from PIL import Image

# Function to preprocess image for ResNet
def preprocess_image_resnet_1(image):
    image = cv2.resize(image, (375, 375))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def preprocess_image_resnet_2(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0)
    return image

def preprocess_image_resnet_3(image):
    image = cv2.resize(image, (200, 200))  # Resize image to 200x200
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to preprocess image for FaceNet
def preprocess_image_facenet(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

# Function to generate embedding
def generate_embedding(model, image, model_type):
    if model_type == 'resnet_1':
        preprocessed_image = preprocess_image_resnet_1(image)
        return model.predict(preprocessed_image)[0]
    if model_type == 'resnet_2':
        preprocessed_image = preprocess_image_resnet_2(image)
        with torch.no_grad():  # No need to track gr100adients
            embedding = model(preprocessed_image)
            return embedding.numpy()[0]
    if model_type == 'resnet_3':
        preprocessed_image = preprocess_image_resnet_3(image)
        return model.predict(preprocessed_image)[0]
    elif model_type == 'facenet':
        preprocessed_image = preprocess_image_facenet(image)
        return model.predict(preprocessed_image)[0]

# Load models
# Supervised ResNet 1
supervised_resnet_1 = tf.keras.models.load_model('supervised_embedding_iman.h5')

# Supervised ResNet 2
supervised_resnet_2_path = 'supervised_embedding_osama.pth'
supervised_resnet_2 = torch.load(supervised_resnet_2_path, map_location=torch.device('cpu'))
supervised_resnet_2.eval()  # Set the model to evaluation mode

# Unsupervised ResNet 
def create_embedding_model():
    base_cnn = resnet.ResNet50(weights="imagenet", input_shape=(200, 200, 3), include_top=False)

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding_model = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return embedding_model

unsupervised_resnet = create_embedding_model()
unsupervised_resnet.load_weights('unsupervised_embedding_jun.h5')

# Unsupervised FaceNet
unsupervised_facenet = FaceNet().model
unsupervised_facenet.load_weights('unsupervised_embedding_iman.h5')

# Load test images and generate embeddings
test_data_path = "/Users/iman/Downloads/Assignment_2/Supervised_model/classification_data/test_data"
num_labels_to_include = 4000
embeddings_resnet_1 = []
embeddings_resnet_2 = []
embeddings_resnet_3 = []
embeddings_facenet = []

labels = []
selected_labels = set()

for folder in os.listdir(test_data_path):
    folder_path = os.path.join(test_data_path, folder)
    if os.path.isdir(folder_path):
        selected_labels.add(folder)
        if len(selected_labels) > num_labels_to_include:
            break
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            embedding_resnet_1 = generate_embedding(supervised_resnet_1, image, 'resnet_1')
            embedding_resnet_2 = generate_embedding(supervised_resnet_2, image, 'resnet_2')
            embedding_resnet_3 = generate_embedding(unsupervised_resnet, image, 'resnet_3')
            embedding_facenet = generate_embedding(unsupervised_facenet, image, 'facenet')

            embeddings_resnet_1.append(embedding_resnet_1)
            embeddings_resnet_2.append(embedding_resnet_2)
            embeddings_resnet_3.append(embedding_resnet_3)
            embeddings_facenet.append(embedding_facenet)
            labels.append(folder)

# Function to create pairs
def create_pairs(embeddings, labels):
    pair_embeddings = []
    pair_labels = []

    label_set = list(set(labels))
    
    # Positive pairs
    for label in label_set:
        same_label_indices = [i for i, l in enumerate(labels) if l == label]
        for i in range(len(same_label_indices)):
            for j in range(i + 1, len(same_label_indices)):
                pair_embeddings.append((embeddings[same_label_indices[i]], embeddings[same_label_indices[j]]))
                pair_labels.append(1)  # Same person

    # Negative pairs
    for i in range(len(pair_labels)):  # Equal number of negative samples
        label1, label2 = random.sample(label_set, 2)
        idx1, idx2 = random.choice([i for i, l in enumerate(labels) if l == label1]), random.choice([i for i, l in enumerate(labels) if l == label2])
        pair_embeddings.append((embeddings[idx1], embeddings[idx2]))
        pair_labels.append(0)  # Different persons

    return pair_embeddings, pair_labels

# Calculate cosine similarity for each pair
def calculate_similarity(pair_embeddings):
    scores = []
    for emb1, emb2 in pair_embeddings:
        score = 1 - cosine(emb1, emb2)  # Cosine similarity
        scores.append(score)
    return scores

# Create pairs and calculate similarity scores for each model
pair_embeddings_resnet_1, pair_labels_resnet_1 = create_pairs(embeddings_resnet_1, labels)
scores_resnet_1 = calculate_similarity(pair_embeddings_resnet_1)

pair_embeddings_resnet_2, pair_labels_resnet_2 = create_pairs(embeddings_resnet_2, labels)
scores_resnet_2 = calculate_similarity(pair_embeddings_resnet_2)

pair_embeddings_resnet_3, pair_labels_resnet_3 = create_pairs(embeddings_resnet_3, labels)
scores_resnet_3 = calculate_similarity(pair_embeddings_resnet_3)

pair_embeddings_facenet, pair_labels_facenet = create_pairs(embeddings_facenet, labels)
scores_facenet = calculate_similarity(pair_embeddings_facenet)

# ROC Analysis for each model
fpr_resnet_1, tpr_resnet_1, _ = roc_curve(pair_labels_resnet_1, scores_resnet_1)
roc_auc_resnet_1 = auc(fpr_resnet_1, tpr_resnet_1)

fpr_resnet_2, tpr_resnet_2, _ = roc_curve(pair_labels_resnet_2, scores_resnet_2)
roc_auc_resnet_2 = auc(fpr_resnet_2, tpr_resnet_2)

fpr_resnet_3, tpr_resnet_3, _ = roc_curve(pair_labels_resnet_3, scores_resnet_3)
roc_auc_resnet_3 = auc(fpr_resnet_3, tpr_resnet_3)

fpr_facenet, tpr_facenet, _ = roc_curve(pair_labels_facenet, scores_facenet)
roc_auc_facenet = auc(fpr_facenet, tpr_facenet)

# Plot ROC Curve for all models
background_color = '#FEFAF6'  # You can choose any color you like

# Plot ROC Curve for all models
plt.figure(facecolor=background_color)
plt.plot(fpr_resnet_1, tpr_resnet_1, color='blue', lw=2, label='ResNet1 ROC curve (area = %0.2f)' % roc_auc_resnet_1)
plt.plot(fpr_resnet_2, tpr_resnet_2, color='red', lw=2, label='ResNet2 ROC curve (area = %0.2f)' % roc_auc_resnet_2)
plt.plot(fpr_resnet_3, tpr_resnet_3, color='orange', lw=2, label='ResNet3 ROC curve (area = %0.2f)' % roc_auc_resnet_3)
plt.plot(fpr_facenet, tpr_facenet, color='green', lw=2, label='FaceNet ROC curve (area = %0.2f)' % roc_auc_facenet)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for All Models')
plt.legend(loc="lower right")

# Set axes background color
ax = plt.gca()  # Get current axes
ax.set_facecolor(background_color)

# Show the plot
plt.show()
