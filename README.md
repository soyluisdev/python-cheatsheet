# **Python Cheat Sheet**

## Comandos Básicos

### Variables y Tipos de Datos


```python
x = 5            # Integer
y = 3.14         # Float
name = "Alice"   # String
is_valid = True  # Boolean
```


```python
# Aritméticos
a + b  # Suma
a - b  # Resta
a * b  # Multiplicación
a / b  # División
a % b  # Módulo
a ** b # Exponenciación

# Comparación
a == b  # Igual a
a != b  # Diferente a
a > b   # Mayor que
a < b   # Menor que
a >= b  # Mayor o igual que
a <= b  # Menor o igual que

# Lógicos
a and b # Y lógico
a or b  # O lógico
not a   # Negación lógica

```

### Estructura de control 


```python
# Condicionales
if condition:
    # código
elif another_condition:
    # código
else:
    # código

# Bucles
for i in range(10):
    # código

while condition:
    # código

```

### Funciones


```python
def function_name(parameters):
    # código
    return value

# Ejemplo de función
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # Output: Hello, Alice!

```

## Built-in functions
### Funciones de tipo y conversión


```python
int(x)        # Convierte a entero
float(x)      # Convierte a float
str(x)        # Convierte a string
bool(x)       # Convierte a booleano
list(iterable) # Convierte a lista
tuple(iterable) # Convierte a tupla
set(iterable) # Convierte a conjunto
dict(mapping) # Convierte a diccionario
```

### Funciones de entrada/salida


```python
print(*objects, sep=' ', end='\n')  # Imprime a la salida estándar
input(prompt)  # Entrada del usuario
```

### Funciones numéricas


```python
abs(x)        # Valor absoluto
round(x[, n]) # Redondea al número de decimales especificado
max(iterable, *[, key, default])  # Mayor valor
min(iterable, *[, key, default])  # Menor valor
sum(iterable, /, start=0)  # Suma de los elementos
```

### Funciones de secuencias


```python
len(s)        # Longitud de una secuencia
sorted(iterable, *, key=None, reverse=False)  # Devuelve una lista ordenada
reversed(seq) # Iterador que devuelve los elementos en orden inverso
enumerate(iterable, start=0) # Devuelve un iterador que produce tuplas (índice, elemento)
zip(*iterables)  # Agrega elementos de iterables
```

### Funciones de mapeo y filtrado


```python
map(function, iterable)   # Aplica una función a cada elemento de un iterable
filter(function, iterable) # Filtra elementos de un iterable
```

### Funciones de objetos y clases


```python
type(object)  # Tipo de objeto
isinstance(object, classinfo)  # Verifica si un objeto es una instancia de una clase
getattr(object, name[, default]) # Obtiene el valor de un atributo de un objeto
setattr(object, name, value)     # Establece el valor de un atributo de un objeto
hasattr(object, name)  # Verifica si un objeto tiene un atributo
delattr(object, name)  # Elimina un atributo de un objeto
```

### Funciones de colecciones


```python
all(iterable)  # Verifica si todos los elementos de un iterable son verdaderos
any(iterable)  # Verifica si algún elemento de un iterable es verdadero
```

### Funciones de archivos


```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)  # Abre un archivo
```

### Funciones de manipulación de objetos


```python
dir([object])  # Lista los atributos de un objeto
id(object)     # Devuelve el identificador del objeto
hash(object)   # Devuelve el valor hash del objeto
help([object]) # Muestra la ayuda sobre el objeto
callable(object) # Verifica si un objeto es llamable
```

### Funciones matemáticas


```python
divmod(a, b)   # Devuelve el cociente y el resto
pow(x, y[, z]) # Potencia con módulo
```

### Funciones avanzadas


```python
eval(expression[, globals[, locals]]) # Evalúa una expresión
exec(object[, globals[, locals]])     # Ejecuta código dinámicamente
```

### Funciones especiales


```python
chr(i)        # Devuelve el carácter Unicode correspondiente
ord(c)        # Devuelve el valor Unicode de un carácter
bin(x)        # Devuelve la representación binaria de un entero
hex(x)        # Devuelve la representación hexadecimal de un entero
oct(x)        # Devuelve la representación octal de un entero
```

## Métodos

### Métodos de cadenas (str)


```python
s.capitalize()   # Capitaliza la primera letra
s.lower()        # Convierte a minúsculas
s.upper()        # Convierte a mayúsculas
s.title()        # Capitaliza la primera letra de cada palabra
s.strip([chars]) # Elimina espacios o caracteres especificados al inicio y final
s.replace(old, new[, count]) # Reemplaza ocurrencias de una subcadena
s.split([sep[, maxsplit]])   # Divide la cadena en una lista
s.join(iterable) # Une una lista de cadenas con la cadena actual como separador
s.find(sub[, start[, end]])  # Devuelve el índice de la primera aparición de una subcadena
s.count(sub[, start[, end]]) # Cuenta las apariciones de una subcadena
s.startswith(prefix[, start[, end]]) # Verifica si la cadena empieza con un prefijo
s.endswith(suffix[, start[, end]])   # Verifica si la cadena termina con un sufijo
```

### Métodos de listas (list)


```python
l.append(x)    # Agrega un elemento al final de la lista
l.extend(iterable) # Extiende la lista con los elementos de un iterable
l.insert(i, x) # Inserta un elemento en una posición específica
l.remove(x)    # Elimina la primera aparición de un elemento
l.pop([i])     # Elimina y devuelve el elemento en la posición especificada
l.clear()      # Elimina todos los elementos de la lista
l.index(x[, start[, end]]) # Devuelve el índice de la primera aparición de un elemento
l.count(x)     # Cuenta las apariciones de un elemento
l.sort(key=None, reverse=False) # Ordena la lista
l.reverse()    # Invierte los elementos de la lista
```

### Métodos de diccionarios (dict)


```python
d.keys()       # Devuelve una vista de las claves del diccionario
d.values()     # Devuelve una vista de los valores del diccionario
d.items()      # Devuelve una vista de los pares clave-valor
d.get(key[, default]) # Devuelve el valor de una clave
d.update([other]) # Actualiza el diccionario con pares clave-valor de otro diccionario
d.pop(key[, default]) # Elimina y devuelve el valor de una clave
d.popitem()    # Elimina y devuelve un par clave-valor arbitrario
d.clear()      # Elimina todos los elementos del diccionario
d.setdefault(key[, default]) # Devuelve el valor de una clave, si no existe la agrega con un valor por defecto
```

### Métodos de conjuntos (set)


```python
s.add(elem)        # Agrega un elemento al conjunto
s.update(*others)  # Actualiza el conjunto con elementos de otros iterables
s.remove(elem)     # Elimina un elemento del conjunto, genera error si no existe
s.discard(elem)    # Elimina un elemento del conjunto, no genera error si no existe
s.pop()            # Elimina y devuelve un elemento arbitrario del conjunto
s.clear()          # Elimina todos los elementos del conjunto
s.union(*others)       # Devuelve la unión de conjuntos
s.intersection(*others) # Devuelve la intersección de conjuntos
s.difference(*others)   # Devuelve la diferencia de conjuntos
s.symmetric_difference(other) # Devuelve la diferencia simétrica de conjuntos
s.isdisjoint(other)     # Verifica si los conjuntos son disjuntos
s.issubset(other)       # Verifica si es un subconjunto
s.issuperset(other)     # Verifica si es un superconjunto
```

### Métodos de tuplas (tuple)


```python
t.count(value)    # Cuenta las apariciones de un valor
t.index(value[, start[, stop]]) # Devuelve el índice de la primera aparición de un valor
```

### Métodos de archivos (file)


```python
f.read([size])    # Lee datos del archivo
f.readline([size]) # Lee una línea del archivo
f.readlines([hint]) # Lee todas las líneas del archivo
f.write(str)      # Escribe una cadena al archivo
f.writelines(lines) # Escribe una lista de líneas al archivo
f.close()         # Cierra el archivo
f.flush()         # Fuerza la escritura de datos en el archivo
f.seek(offset[, whence]) # Mueve la posición del cursor en el archivo
f.tell()          # Devuelve la posición actual del cursor en el archivo
f.truncate([size]) # Trunca el archivo a un tamaño específico
```

## Principales librerías

### Librerías estándar

#### math - Funciones Matemáticas


```python
import math

math.sqrt(16)       # Raíz cuadrada
math.sin(math.pi/2) # Seno
math.cos(math.pi)   # Coseno
math.log(10)        # Logaritmo natural
```

#### datetime - Manipulación de Fechas y Horas


```python
import random

random.random()    # Número aleatorio entre 0 y 1
random.randint(1, 10) # Entero aleatorio entre 1 y 10
random.choice(['a', 'b', 'c']) # Elección aleatoria de una lista
random.shuffle([1, 2, 3, 4, 5]) # Mezcla aleatoria de una lista

```

#### random - Generación de Números Aleatorios


```python
import random

random.random()    # Número aleatorio entre 0 y 1
random.randint(1, 10) # Entero aleatorio entre 1 y 10
random.choice(['a', 'b', 'c']) # Elección aleatoria de una lista
random.shuffle([1, 2, 3, 4, 5]) # Mezcla aleatoria de una lista
```

#### os - Interacción con el Sistema Operativo


```python
import os

os.getcwd()        # Obtener el directorio de trabajo actual
os.listdir('.')    # Listar archivos en el directorio actual
os.mkdir('new_dir')# Crear un nuevo directorio
os.remove('file.txt') # Eliminar un archivo
```

#### sys - Parámetros y Funciones del Intérprete


```python
import sys

sys.argv           # Lista de argumentos de la línea de comandos
sys.exit()         # Salir del programa
sys.path           # Lista de rutas de búsqueda de módulos
```

---

### Librerías externas

#### numpy - Computación Numérica


```python
import numpy as np

arr = np.array([1, 2, 3, 4])   # Crear un array
zeros = np.zeros((3, 3))       # Array de ceros
ones = np.ones((3, 3))         # Array de unos
mean = np.mean(arr)            # Media de los elementos
```

#### pandas - Manipulación de Datos


```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}) # Crear un DataFrame
df.head()         # Primeras 5 filas del DataFrame
df.describe()     # Resumen estadístico
df['a']           # Seleccionar una columna
df.loc[0]         # Seleccionar una fila por etiqueta
```

#### matplotlib - Visualización de Datos


```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6]) # Crear un gráfico de línea
plt.title('Título')            # Agregar un título
plt.xlabel('Eje X')            # Etiquetar el eje X
plt.ylabel('Eje Y')            # Etiquetar el eje Y
plt.show()                     # Mostrar el gráfico
```

#### scipy - Computación Científica 


```python
from scipy import stats

data = [1, 2, 3, 4, 5]
mean = stats.tmean(data)       # Media truncada
std_dev = stats.tstd(data)     # Desviación estándar truncada
```

#### requests - Solicitudes HTTP


```python
import requests

response = requests.get('https://api.example.com/data')
data = response.json()         # Obtener respuesta en formato JSON
response.status_code           # Código de estado de la respuesta
```

#### flask - Desarrollo de Aplicaciones Web


```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()

```

#### django - Desarrollo Web a Gran Escala


```python
# En el archivo settings.py
INSTALLED_APPS = [
    # ...
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

# En el archivo views.py
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, world!")

```

#### pytest - Pruebas


```python
# En el archivo settings.py
INSTALLED_APPS = [
    # ...
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

# En el archivo views.py
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, world!")

```

#### beautifulsoup4 - Análisis de Documentos HTML y XML


```python
from bs4 import BeautifulSoup

html_doc = "<html><head><title>The Dormouse's story</title></head><body><p>Once upon a time...</p></body></html>"
soup = BeautifulSoup(html_doc, 'html.parser')
title = soup.title.string  # 'The Dormouse's story'

```

### Librerías de Machine Learning

#### scikit-learn - Machine Learning


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

#### tensorflow - Deep Learning


```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5)
```

#### pytorch - Deep Learning


```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Dummy training loop
for epoch in range(2):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Comandos de Lectura de Archivos


### Lectura Básica de Archivos

#### Abrir y Leer Archivos


```python
# Abrir un archivo para lectura
file = open('file.txt', 'r')

# Leer todo el contenido del archivo
content = file.read()

# Leer una línea del archivo
line = file.readline()

# Leer todas las líneas del archivo en una lista
lines = file.readlines()

# Cerrar el archivo
file.close()
```

#### Uso de Context Manager


```python
# Usar 'with' para manejar archivos automáticamente
with open('file.txt', 'r') as file:
    content = file.read()

# El archivo se cierra automáticamente al salir del bloque 'with'
```

### Lectura de Archivos Línea por Línea

#### Usar un Bucle para Leer Líneas


```python
with open('file.txt', 'r') as file:
    for line in file:
        print(line.strip())  # .strip() elimina los caracteres de nueva línea
```

#### Leer un Número Específico de Caracteres


```python
with open('file.txt', 'r') as file:
    chunk = file.read(100)  # Leer los primeros 100 caracteres
```

### Lectura de Archivos CSV

#### Usar el Módulo csv


```python
import csv

# Leer un archivo CSV
with open('file.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Leer un archivo CSV con encabezados
with open('file.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['column_name'])
```

### Lectura de Archivos JSON

#### Usar el Módulo json


```python
import json

# Leer un archivo JSON
with open('file.json', 'r') as file:
    data = json.load(file)
    print(data)
```

### Lectura de Archivos XML

#### Usar el Módulo xml.etree.ElementTree


```python
import xml.etree.ElementTree as ET

# Leer un archivo XML
tree = ET.parse('file.xml')
root = tree.getroot()

for child in root:
    print(child.tag, child.attrib)
```

### Lectura de Archivos Excel

#### Usar el Módulo pandas


```python
import pandas as pd

# Leer un archivo Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
print(df.head())
```

### Lectura de Archivos ZIP

#### Usar el Módulo zipfile


```python
import zipfile

# Leer archivos dentro de un archivo ZIP
with zipfile.ZipFile('file.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_folder')

# Leer un archivo específico dentro del ZIP sin extraerlo
with zipfile.ZipFile('file.zip', 'r') as zip_ref:
    with zip_ref.open('file_inside_zip.txt') as file:
        content = file.read()
        print(content)
```

### Lectura de Archivos de Texto con pathlib

#### Usar el Módulo pathlib


```python
from pathlib import Path

# Leer un archivo usando Path
file_path = Path('file.txt')
content = file_path.read_text()
print(content)

# Leer un archivo binario usando Path
binary_content = file_path.read_bytes()
print(binary_content)
```

# Design Patterns
## 1. Creational  
1. Singleton
2. Factory Method
3. Builder
4. Prototype
5. Abstract Factory  
## 2. Structural  
1. Adaptaer
2. Decorator
3. Facade
4. Composite
5. Flyweight
6. Bridge  
## 3. Behavioral  
1. Observer
2. Commnad
3. Strategy
4. State
5. Chan of Responsibility

## Creational   
### Singleton


```python


class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)

        return cls._instances[cls]

obj1 = Singleton()
obj2 = Singleton()

print(obj1 is obj2)

import sqlite3

class DatabaseConnection(Singleton):
    connection = None

    def __init__(self):
        if self.connection is None:
            self.connection = sqlite3.connect("users.db")

    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        self.connection.commit()

    def close(self):
        self.connection.close()


db1 = DatabaseConnection()
db1.execute_query("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")

db2 = DatabaseConnection()
db2.execute_query("INSERT INTO users (name) VALUES ('Name')")

print(db1 is db2)

db1.close()
db2.close()
```

    True
    True



```python

```
