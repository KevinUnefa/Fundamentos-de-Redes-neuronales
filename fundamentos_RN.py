""" Función General del Código
El objetivo principal de este script es construir y entrenar un "cerebro" digital
(una Red Neuronal Artificial) capaz de reconocer y clasificar imágenes de dígitos 
escritos a mano del 0 al 9. """

##python
import tensorflow as tf     # TensorFlow es una biblioteca de código abierto creada y manteida por Google para el desarrollo de IA's,  aprendizaje automático (machine learning), aprendizaje profundo. Sirve para construir y entrenar redes neuronales y otros modelos de IA
                            # Keras: es una API (Interfaz de Programación de Aplicaciones) de alto nivel dentro de TensorFlow. Es la parte que nos permite definir modelos, capas y procesos de entrenamiento de forma muy sencilla e intuitiva
from tensorflow.keras.datasets import mnist   # datasets es el conjunto de datos de Keras, y mnist es un objeto que contiene 70,000 imágenes de dígitos manuscritos (60,000 para entrenar, 10,000 para probar).
from tensorflow.keras.models import Sequential  # extraer un modelo especifco con el cual vamos a entrenar nuestra Red. El modelo Sequential es una clase de modelo que permite crear una arquitectura de red neuronal como una simple "pila de capas", una seguida de la otra.
from tensorflow.keras.layers import Dense, Flatten # se utiliza para cambiar la forma (dimensiones) de los datos de entrada, convirtiendo un tensor multidimensional en un tensor unidimensional. Es una capa totalmente conectada, es decir, cada neurona de esta capa está conectada a todas las neuronas de la capa anterior
from tensorflow.keras.utils import to_categorical  # Esta función convierte nuestras etiquetas numéricas en un formato llamado "one-hot encoding". Convierte un vector de etiquetas de clase enteras en una matriz binaria. Ej: 2 = [0, 0, 1]

# Cargar datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Crear modelo sencillo, establece las reglas de aprendizaje
model = Sequential([
 Flatten(input_shape=(28, 28)), 
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')
])

# Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluar
model.evaluate(x_test, y_test)