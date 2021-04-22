import tensorflow as tf

train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
print(images.shape)
print(labels.shape)
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset.element_spec)