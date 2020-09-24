import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from inputDatagen import circuitgen as crg
#
n = 2
m = 10000

dataClass = crg(total_layer = n,total_instance = m)
data = dataClass.getInputData()
##print(data)

df = pd.DataFrame(data)
X = np.array(df.drop(['epsilon'], 1))
y = np.array(df['epsilon'])

X = preprocessing.scale(X)
y = np.array(df['epsilon'])


model = keras.models.Sequential([
                                 keras.layers.Dense(8*n, activation='linear'),
                                 keras.layers.Dense(246, activation="relu",kernel_initializer="he_normal"),
                                 keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
                                 keras.layers.Dense(1, activation='linear')
                                 ])

model.compile(loss="mse", optimizer='adam',
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(X_train,y_train, epochs=20)
