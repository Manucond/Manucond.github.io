import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset cargado correctamente:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

# Análisis exploratorio
def exploratory_analysis(data):
    print("\nEstadísticas básicas:")
    print(data.describe())
    
    print("\nValores nulos por columna:")
    print(data.isnull().sum())
    
    # Visualización básica
    data.hist(bins=20, figsize=(10, 8))
    plt.suptitle("Distribución de datos")
    plt.show()

# Modelo de predicción
def predict(data, input_col, output_col):
    # Dividir datos
    X = data[[input_col]]
    y = data[output_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    
    print("\nResultados de predicción:")
    print(f"Coeficiente: {model.coef_[0]}")
    print(f"Intercepto: {model.intercept_}")
    print(f"Error cuadrático medio (MSE): {error}")
    
    # Visualización de predicciones
    plt.scatter(X_test, y_test, color='blue', label="Valores reales")
    plt.plot(X_test, y_pred, color='red', label="Predicciones")
    plt.xlabel(input_col)
    plt.ylabel(output_col)
    plt.legend()
    plt.title("Regresión Lineal")
    plt.show()

# Programa principal
if __name__ == "__main__":
    # Ruta del archivo (puedes cambiar este archivo para probar con tus datos)
    file_path = "data.csv"
    
    data = load_dataset(file_path)
    if data is not None:
        exploratory_analysis(data)
        
        # Cambia las columnas aquí según tu dataset
        input_column = "X"  # Por ejemplo: edad, horas trabajadas, etc.
        output_column = "Y"  # Por ejemplo: salario, puntuación, etc.
        
        if input_column in data.columns and output_column in data.columns:
            predict(data, input_column, output_column)
        else:
            print("Asegúrate de que las columnas especificadas existan en el dataset.")
