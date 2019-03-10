from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
#Only show tensorflow errors, to avoid deprecation warnings
tf.logging.set_verbosity(tf.logging.ERROR)

#Fixed random seed so tuning is easier
np.random.seed(7)

#Load dataset, assuming the first row contains labels
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",", skiprows=1)

#Splitting into into input (X) and output (Y) variables
#Take all first columns except the last
X = dataset[:,0:-1]
#Take the last column
Y = dataset[:,-1]

print("Dataset length: %s" % (len(X)))

#Creating the model
model = Sequential()
model.add(Dense(4, input_dim=len(X[0]), activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fit the model
model.fit(X, Y, epochs=1500, batch_size=100)

#Evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(model.predict(np.array([[6,148,72,35,0,33.6,0.627,50]])))
