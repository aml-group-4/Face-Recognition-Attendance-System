import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Layer, Input
from keras.preprocessing import image
from keras_facenet import FaceNet
import numpy as np
import matplotlib.pyplot as plt

# Redefine the DistanceLayer (if used in your Siamese model)
class DistanceLayer(Layer):
    """
    Computes the distance between anchor, positive, and negative embeddings.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

# Build the embedding model
def build_embedding_model():
    facenet = FaceNet()
    keras_model = facenet.model  # Access the Keras model in FaceNet

    embedding_model = Model(inputs=keras_model.input, 
                            outputs=keras_model.layers[-2].output,  # Adjust based on the correct layer
                            name="Embedding")
    return embedding_model

embedding_model = build_embedding_model()
embedding_model.load_weights("unsupervised_embedding.h5")

# Define the preprocess function
def preprocess_image(filename, target_shape):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    
    return image

# Generate an embedding for an image
def generate_embedding(model, image_path):
    """
    Generate an embedding for the given image using the specified model.
    """
    img = preprocess_image(image_path, target_shape=(160, 160))  # Adjust size if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    embedding = model.predict(img)
    return embedding

# Example usage
image_path = 'iman.jpeg'  # Replace with your image path
embedding = generate_embedding(embedding_model, image_path)

# Display the generated embedding
print("Generated Embedding:", embedding)

# Optionally, visualize the embedding
plt.plot(embedding.flatten())
plt.title("Embedding Visualization")
plt.xlabel("Dimensions")
plt.ylabel("Values")
plt.show()
