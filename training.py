# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Training a tensorflow model to predict heating and cooling load based on building features.
import numpy
import pandas
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Parameters.
batchSize = 10
epochs = 500

# Training data url (https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx).

# Get data from url.
data = pandas.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx")
data = data.sample(frac=1).reset_index(drop=True)

# Split data into training and validation (90/10).
training , validation = train_test_split(data, test_size=0.2)
training_stats = training.describe()

# Get two outputs and save stats.
training_stats.pop('Y1')
training_stats.pop('Y2')
training_stats = training_stats.transpose()

# Format training outputs.
training_output = numpy.array(training.pop("Y1")) , numpy.array(training.pop("Y2"))

# Format validation outputs.
validation_output = numpy.array(validation.pop("Y1")) , numpy.array(validation.pop("Y2"))

# Normalize training and validation data.
training_data = (training - training_stats['mean']) / training_stats['std']
validation_data = (validation - training_stats['mean']) / training_stats['std']

# Define layers.
input_layer = keras.layers.Input(shape=(8,))
first_dense_layer = keras.layers.Dense(units=128 , activation="relu")(input_layer)
second_dense_layer = keras.layers.Dense(units=128, activation="relu")(first_dense_layer)
third_dense_layer = keras.layers.Dense(units=64, activation="relu")(second_dense_layer)

# Output one from second layer.
output_one = keras.layers.Dense(units=1, name="output_one")(second_dense_layer)
# Output two from third layer.
output_two = keras.layers.Dense(units=1, name="output_two")(third_dense_layer)

# Create a model.
model = keras.models.Model(inputs=input_layer, outputs=[output_one, output_two])

# Set loss function and optimizer.
model.compile(loss={"output_one" :"mse" , "output_two" : "mse"},
              optimizer="adam")


# Train model.
model.fit(training_data, training_output , epochs=epochs, batch_size=batchSize , validation_data=(validation_data, validation_output) , verbose=1)