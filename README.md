# Fundamentos-de-Redes-neuronales

## Función General del Código
El objetivo principal de este script es construir y entrenar un "cerebro" digital (una Red Neuronal Artificial) capaz de reconocer y clasificar imágenes de dígitos escritos a mano del 0 al 9.
Para lograrlo, el código sigue un flujo de trabajo estándar en Machine Learning:
1.  Reúne los materiales: Importa las herramientas (librerías) y los datos (imágenes MNIST).
2.	Prepara los datos: Limpia y formatea las imágenes y sus etiquetas.
3.	Diseña el "cerebro": Define la arquitectura de la red neuronal, modelo, capas y funcion de activación.
4.	Establece las reglas de aprendizaje: Le dice al modelo cómo debe aprender.
5.	Entrena: Alimenta al modelo con los datos de entrenamiento para que aprenda a asociar imágenes con etiquetas.
6.	Evalúa: Prueba el modelo con datos que nunca ha visto para ver qué tan bien aprendió.

## Importación de Librerías
Importamos las herramientas que necesitas antes de empezar a construir.

- import tensorflow as tf: TensorFlow es una biblioteca de código abierto creada y manteida por Google para el desarrollo de IA's,  aprendizaje automático (machine learning), aprendizaje profundo. Sirve para construir y entrenar redes neuronales y otros modelos de IA.
- Keras: es una API de alto nivel dentro de TensorFlow. Es la parte que nos permite definir modelos, capas y procesos de entrenamiento de forma muy sencilla e intuitiva
- Datasets y MNIST: datasets es el conjunto de datos de Keras, y mnist es un objeto que contiene 70,000 imágenes de dígitos manuscritos (60,000 para entrenar, 10,000 para probar).
- Modelos y Sequential: tensorflow.keras.models se utiliza para extraer un modelo especifco con el cual vamos a entrenar nuestra RNA. El modelo Sequential es una clase de modelo que permite crear una arquitectura de red neuronal como una simple "pila de capas", una seguida de la otra.
- Keras.layers: Las capas (o layers) de Keras son los bloques de construcción fundamentales de los modelos de redes neuronales. Cada capa es un módulo de procesamiento de datos que recibe información de entrada (como uno o más tensores), realiza una operación o cálculo específico sobre ella y genera una salida.
- Flatten: se utiliza para "aplanar" los datos de entrada, convirtiendo una matriz multidimensional en una matriz unidimensional (un vector plano). En este caso, las imágenes vienen como matrices de 28x28 píxeles que es una matriz 2D, la capa Flatten la convierte en un vector de 784 elementos (28 * 28).
- Dense: Es una capa totalmente conectada (o densa), lo que significa que cada neurona de esta capa está conectada a todas las neuronas de la capa anterior.
- tf.keras.utils: El módulo tensorflow.keras.utils proporciona una colección de funciones de utilidad y herramientas auxiliares.
- to_categorical: Esta función convierte nuestras etiquetas numéricas en un formato llamado "one-hot encoding". Convierte un vector de etiquetas de clase enteras en una matriz binaria (de ceros y unos). Sirve para preparar los datos de etiquetas (las "salidas" o targets) para modelos de deep learning que resuelven problemas de clasificación multiclase.


## Cargar y Preparar Datos
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
mnist.load_data(): Esta función ejecuta el dataset que importamos. Y le asigna las imagenes la sus respectivas variables
- (x_train, y_train): El conjunto de entrenamiento. x_train son las 60,000 imágenes (los inputs) y y_train son sus 60,000 etiquetas correspondientes (los outputs o la "verdad"). El modelo aprenderá de estos datos.
- (x_test, y_test): El conjunto de prueba. x_test son las 10,000 imágenes y y_test sus 10,000 etiquetas. Estos datos se usan al final para evaluar qué tan bien generaliza el modelo.

x_train, x_test = x_train / 255.0, x_test / 255.0
¿A que se debe esta operación?, bueno esto se llama normalización. Las imágenes originales tienen valores de píxel que van de 0 (negro) a 255 (blanco). Dividir entre 255.0 escala todos los valores de los píxeles para que queden en el rango [0, 1].
¿Por qué? Las redes neuronales aprenden mucho más rápido y de manera más estable cuando los datos de entrada tienen valores pequeños y de rangos similares.

y_train, y_test = to_categorical(y_train), to_categorical(y_test)
Lo transformamos en One-Hot Encoding.
¿Por qué? Esta red tendrá 10 neuronas de salida (una para cada dígito). Si la respuesta correcta es "7", 8va neurona (índice 7) se active y las demás no. El formato [0,0,0,0,0,0,0,1,0,0] le dice al modelo exactamente esto. Es el formato que la función de pérdida categorical_crossentropy espera.


##Crear Modelo Sencillo
Creamos una instancia de nuestro modelo en este caso llamado model, indicando el tipo de modelo que vamos a usar, en este caso Sequential. Le pasamos una lista [...] de las capas que queremos, serían los parámetros, en el orden en que los datos fluirán a través de ellas.

Capa 1: Flatten(input_shape=(28, 28))
La capa Flatten se utiliza para transformar los datos de entrada multidimensionales (en este caso una imagen 2D) en un vector unidimensional (aplanado). 
Parámetro clave: input_shape este es un parámetro crucial en la primera capa del modelo, ya que define las dimensiones de los datos de entrada que la red espera procesar En este caso, indica que cada entrada es una imagen de 28x28 píxeles. Tomará la matriz de (28, 28) píxeles y la convertirá en un vector plano de 28 * 28 = 784 neuronas.

Capa 2: Dense(128, activation='relu') 
Esta sería la primera y única capa oculta. En los parametros tenemos:
- Primer argumento (128) se refiere a las unidades units (unidades o neuronas) define el número de neuronas en esa capa oculta. Cada una de estas 128 neuronas tiene sus propios pesos y 		sesgos que se aprenden durante el entrenamiento. La elección del número de neuronas afecta la capacidad del modelo para aprender patrones complejos.
- Segundo argumento: activation='relu' (Rectified Linear Unit): Especifica la función de activación que se aplicará a la salida de esta capa. La función relu introduce no linealidad en 		el modelo, su funcion principal es, si la entrada es negativa, la salida es 0; si es positiva, la salida es la misma entrada. Permite que la red aprenda relaciones más complejas que 		las simples transformaciones lineales. 

Capa 3: Dense(10, activation='softmax')
Esta es la capa de salida del modelo, también una capa Dense. Que se le atribuye los siguientes parametros:
- 10 neuronas como parámetro de salida, lo cual es apropiado para un problema de clasificación con 10 clases posibles, en este caso los dígitos del 0 al 9 del conjunto de datos MNIST.
- Funcion de activación: activation valor = 'softmax', esta función de activación se utiliza comúnmente en la capa de salida para problemas de clasificación multiclase. softmax convierte las salidas numéricas de la capa en una distribución de probabilidad, donde la suma de todas las probabilidades para las 10 clases es igual a 1. La clase con la probabilidad más alta es la predicción del modelo.


## 4. Compilar y Entrenar
Ahora que el modelo está diseñado, le decimos cómo aprender y luego le damos los datos para que lo haga.
El método model.compile(...): configura el proceso de aprendizaje. Pasándole como parámetros un Optimizador (optimizer), Funcion de perdida (loss), y la Métricas (metrics).
- El optimizador es el algoritmo que ajusta los pesos (las conexiones) de la red para minimizar el error. Adam es un optimizador muy popular, su característica principal es que adaptan 		automáticamente la tasa de aprendizaje para cada parámetro del modelo, lo que los hace eficientes para problemas complejos y grandes conjuntos de datos.
- La función de pérdida (loss). Esta es la fórmula que mide qué tan equivocada está la predicción del modelo en cada paso. El objetivo del optimizador es minimizar este valor. 						Categorical_crossentropy es la función de pérdida estándar para clasificación multiclase con etiquetas en formato one-hot.
- Las métricas se utilizan para comparar los resultados de diferentes algoritmos y monitorear el rendimiento. accuracy es el porcentaje de veces que el modelo acertó.

El comando model.fit(...) es el que inicia el entrenamiento.
x_train, y_train son los datos que va a usar para aprender. Estos son las 60,000 imágenes y sus etiquetas.
epochs=5: Una epoch (época) es una pasada completa por todo el conjunto de datos de entrenamiento. Es como decirle "revise los 60,000 ejemplos 5 veces".
validation_split=0.1: Esto es muy útil. Antes de entrenar, separe automáticamente el 10% de los datos de entrenamiento (6,000 imágenes) y los use como un conjunto de validación mientras este aprende.
Al final de cada época, el modelo probará su rendimiento en este 10% (el set de validación). Esto nos permite ver si el modelo está sobre ajustando, es decir, si se está volviendo muy bueno con los datos de entrenamiento, pero malo con datos nuevos.


## 5. Evaluar
model.evaluate(x_test, y_test)
Usamos el conjunto de prueba (x_test, y_test) que el modelo nunca ha visto. 
Este método calcula la pérdida y la precisión (y cualquier otra métrica que hayamos compilado) en un conjunto de datos.

x_test, y_test: Son los 10,000 ejemplos de prueba.
El resultado de esto es la verdadera medida del rendimiento del modelo. "Pérdida: 0.08, Precisión: 0.975".
