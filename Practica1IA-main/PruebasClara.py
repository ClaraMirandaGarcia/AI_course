import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from six import StringIO 
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Directorio base y prefijo
base_dir = 'ManiobrasSimulador'
prefix = 'STISIMData_'
fichero = 'Overtaking.xlsx'

datos = []
step = 20
preproc = []

# Maniobra data almacena todos los pasos que componen un adelantamiento
maniobra_data = []
maniobra_target = []
maniobra = []
es_maniobra = []

try:
    num_drivers = os.listdir(base_dir)
    for i, driver in enumerate(num_drivers, start=1):
        archivo = Path(base_dir) / driver / (prefix  + fichero)
        excl = pd.read_excel(archivo)

        # Realizamos un primer filtrado, para quedarnos solo con las columnas que nos interesen
        proc1 = pd.DataFrame({
            'MomentoTemporal': excl['Elapsed time'],
            'AnguloVolante': excl['Steering wheel angle'],
            'Acelerador': excl['Gas pedal'],
            'Freno': excl['Brake pedal'],
            'Embrague': excl['Clutch pedal'],
            'Marcha': excl['Gear'],
            'Velocidad': excl['speed'],
            'RPM': excl['RPM'],
            'FlagManiobra': excl['Maneuver marker flag']
        })
        #datos.append(proc1)
        var_volante = []
        var_aclrdr = []
        var_freno = []
        var_embrague = []
        var_speed = []
        var_rpm = []

        marchas_inicio = []
        marchas_fin = []
        maniobra = []
        
        arr = proc1.to_numpy()
        flag = proc1['FlagManiobra'].to_numpy()

        for i in range(0, len(proc1), step):
            print("PREPROCESANDO ")
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
        arr = dts_preproc.to_numpy()
            

        # todos los datos excepto el de flag de maniobra que lo guardamos en maniobra_target
        print(len(arr))
        for i in range(len(arr)):
            maniobra = [arr[i][0], int(arr[i][1]), int(arr[i][2]), int(arr[i][3]), arr[i][4], arr[i][5],
                int(arr[i][6]), int(arr[i][7])]
            es_maniobra = int(arr[i][8])
            maniobra_data.append(maniobra)
            maniobra_target.append(es_maniobra)

except Exception as e:
    print(f"Error: {e}")

# Definir las características (X) y la variable objetivo (y)
X = maniobra_data
y = maniobra_target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)  # 70% entrenamiento y 30% prueba

# Crear el clasificador SVM
svm_classifier = SVC(kernel='linear', random_state=42)  # Selecciona el kernel apropiado según la naturaleza de tus datos

# Entrenar el modelo SVM
print("ENTRENANDO MODELO SVM")
svm_classifier.fit(X_train, y_train)
print("FINALIZANDO EL MODELO SVM")
# Predecir las etiquetas para el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Evaluar la precisión del modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM:", accuracy)


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
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()
    

# Visualizar los límites de decisión del modelo SVM
plot_decision_boundary(svm_classifier, X, y)
print("ji")




