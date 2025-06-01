# Practica3IA
Para el correcto funcionamiento del código, es necesario indicar la nueva ruta de los datos de entrenamiento y de test.
En cada uno de los Process Documents from files, se tiene que volver a introducir la ruta a los conjuntos de textos *bbc-train* y *bbc-test*, dependiendo del que se emplee en cada elemento.
Además de estos, también se tiene que volver a indicar la ruta al fichero *diccionario.txt* al ***Filter Stopwords (Dictionary)*** dentro del ***Loop Collection*** del *ModeloNoSupervisadoWordVec*. Dentro de este mismo modelo, puede ser necesario también redirigir al elemento ***Store*** la nueva dirección de Word2Vec\_v1.iooo, o deshabilitar dicho elemento, para asegurar el funcionamiento sin errores del modelo.
