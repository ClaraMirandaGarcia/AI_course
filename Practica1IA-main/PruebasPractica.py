import pandas as pd
import numpy as np
import sklearn as sk
import pydotplus
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation 
from six import StringIO 
from sklearn.tree import plot_tree


from IPython.display import Image  
from sklearn.tree import export_graphviz


prefix = 'ManiobrasSimulador/Driver'
fichero = 'STISIMData_Overtaking.xlsx' 

datos = []

# Leemos los datos de los 5 conductores que se nos pasan como Datasets
for i in range(5):
    indice = i+1

    nombre = prefix + str(indice) + "/" + fichero
    excl = pd.read_excel(nombre)

    # Realizamos un primer filtrado, para quedarnossolo con las columnas que nos interesen
    proc1 = pd.DataFrame({'MomentoTemporal': excl['Elapsed time'],
                          'AnguloVolante': excl['Steering wheel angle'],
                          'Acelerador': excl['Gas pedal'],
                          'Freno': excl['Brake pedal'],
                          'Embrague': excl['Clutch pedal'],
                          'Marcha': excl['Gear'],
                          'Velocidad': excl['speed'],
                          'RPM': excl['RPM'],
                          'FlagManiobra': excl['Maneuver marker flag']})
    datos.append(proc1)

step = 20
preproc = []

for elem in datos:
    var_volante = []
    var_aclrdr = []
    var_freno = []
    var_embrague = []
    var_speed = []
    var_rpm = []

    marchas_inicio = []
    marchas_fin = []
    maniobra = []
    
    arr = elem.to_numpy()
    flag = elem['FlagManiobra'].to_numpy()
    for i in range(0, len(elem), step):
        segundo = arr[i:i+step-1]
        inicio = segundo[0]
        fin = segundo[-1]

        var_volante.append(fin[1] - inicio[1])
        var_aclrdr.append(int(fin[2] - inicio[2]))
        var_freno.append(int(fin[3] - inicio[3]))
        var_embrague.append(int(fin[4] - inicio[4]))
        var_speed.append(fin[6] - inicio[6])
        var_rpm.append(fin[7] - inicio[7])

        marchas_inicio.append(int(inicio[5]))
        marchas_fin.append(int(fin[5]))
        maniobra.append(int(1 in flag[i:i+step]))
    
    dts_preproc = pd.DataFrame({'VariacionVolante': var_volante,
                            'VariacionAcelerador': var_aclrdr,
                            'VariacionFreno': var_freno,
                            'VariacionEmbrague': var_embrague,
                            'VariacionVelocidad': var_speed,
                            'VariacionRPM': var_rpm,
                            'MarchaInicio': marchas_inicio,
                            'MarchaFinal': marchas_fin,
                            'Maniobra': maniobra
    })
    preproc.append(dts_preproc)

# Maniobra data almacena todos los pasos que componen un adelantamiento
maniobra_data = []
maniobra_target = []

# maniobra_data
#   maniobra
#       paso
for elem in preproc:
    arr = elem.to_numpy()
    maniobra = []
    # todos los datos excepto el de flag de maniobra que lo guardamos en maniobra_target
    paso = [arr[0][0], int(arr[0][1]), int(arr[0][2]), int(arr[0][3]), arr[0][4], arr[0][5],
            int(arr[0][6]), int(arr[0][7])]
    maniobra.append(paso)

    # Flag para saber si cambiamos de estado en el Flag de Maniobra
    es_maniobra = int(arr[0][8])
    
    for i in range(1,len(elem)):
        paso = [arr[i][0], int(arr[i][1]), int(arr[i][2]), int(arr[i][3]), arr[i][4], arr[i][5],
            int(arr[i][6]), int(arr[i][7])]
        
        if es_maniobra != int(arr[i][8]):
            
            maniobra_data.append(maniobra)
            # tenemos que tener datos 1 y 0
            maniobra_target.append(es_maniobra)

            es_maniobra = int(arr[i][8])
            maniobra = []
        
        maniobra.append(paso)
    
    #if es_maniobra:
    maniobra_data.append(maniobra)
    maniobra_target.append(es_maniobra)

min_pasos = 100
max_pasos = 0
for elem in maniobra_data:
    min_pasos = min(min_pasos, len(elem))
    max_pasos = max(max_pasos, len(elem))
# cambio el número de pasos en maniobra_data para que sean consistentes
# 8 en cada (es el mínimo para un adelantamiento)

aux = []
for elem in maniobra_data:
    # Selecting the minimum number of data points for consistency
    section = elem[:min_pasos]

    # Reshape the section into a list of lists -> 2D instead of 3D
    section = np.array(section)
    reshaped_section = section.reshape(8*min_pasos)
    aux.append(reshaped_section)

maniobra_data = np.array(aux)

# Definir las características (X) y la variable objetivo (y)
X = maniobra_data
y = np.array(maniobra_target)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% entrenamiento y 30% prueba




# Crear el clasificador SVM
svm_classifier = SVC(kernel='linear')  # Selecciona el kernel apropiado según la naturaleza de tus datos

# Entrenar el modelo SVM
print("ENTRENANDO MODELO SVM")
svm_classifier.fit(X_train, y_train)
print("FINALIZANDO EL MODELO SVM")
# Predecir las etiquetas para el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Evaluar la precisión del modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM:", accuracy)

# Entrenar el clasificador SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Función para visualizar los límites de decisión del SVM en un espacio bidimensional
def plot_decision_boundary(clf, X, y):
    # Definir el rango de valores para los ejes x e y
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Crear una malla de puntos para evaluar el modelo
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Realizar predicciones para cada punto en la malla
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Visualizar los límites de decisión y los puntos de datos
    '''
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()
    '''

# Visualizar los límites de decisión del modelo SVM
plot_decision_boundary(svm_classifier, X, y)


# Crear el objeto del árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo del árbol de decisión
clf = clf.fit(X_train, y_train)

# Predecir las respuestas para el conjunto de datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
print("Precisión:", metrics.accuracy_score(y_test, y_pred))

# Visualizar el árbol de decisión
'''
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=["VariacionVolante", "VariacionAcelerador", "VariacionFreno", "VariacionEmbrague", "VariacionVelocidad", "VariacionRPM", "MarchaInicio", "MarchaFinal"], class_names=['0', '1'])
plt.show()
'''

# OPTIMIZACION DEL ARBOL

# Crear el objeto del árbol de decisión
clf = DecisionTreeClassifier(max_depth=5)

# Entrenar el modelo del árbol de decisión
clf = clf.fit(X_train, y_train)

# Predecir las respuestas para el conjunto de datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
print("Precisión:", metrics.accuracy_score(y_test, y_pred))

# Visualizar el árbol de decisión
'''
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=["VariacionVolante", "VariacionAcelerador", "VariacionFreno", "VariacionEmbrague", "VariacionVelocidad", "VariacionRPM", "MarchaInicio", "MarchaFinal"], class_names=['0', '1'])
plt.show()
'''