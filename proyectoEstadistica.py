import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import math

class ProyectoEstadistica:
    def __init__(self):
        self.datos = None
        self.cargar_datos()
    
    def cargar_datos(self):
        """Carga datos desde un archivo CSV o permite ingresarlos manualmente"""
        print("=" * 50)
        print("PROYECTO PYTHON PARA ANÁLISIS ESTADÍSTICO")
        print("=" * 50)
        
        opcion = input("¿Desea cargar datos desde un archivo CSV? (s/n): ").lower()
        
        if opcion == 's':
            try:
                archivo = input("Ingrese el nombre del archivo CSV: ")
                self.datos = pd.read_csv(archivo)
                print(f"Datos cargados exitosamente. Dimensiones: {self.datos.shape}")
            except FileNotFoundError:
                print("Archivo no encontrado. Se procederá a ingresar datos manualmente.")
                self.ingresar_datos_manual()
        else:
            self.ingresar_datos_manual()
    
    def ingresar_datos_manual(self):
        """Permite al usuario ingresar datos manualmente"""
        print("\nIngreso de datos manual")
        print("Ingrese los números separados por comas (ej: 1,2,3,4,5)")
        
        while True:
            try:
                entrada = input("Datos: ")
                lista_datos = [float(x.strip()) for x in entrada.split(',')]
                self.datos = pd.DataFrame(lista_datos, columns=['Valores'])
                print(f"Datos ingresados: {lista_datos}")
                break
            except ValueError:
                print("Error: Asegúrese de ingresar solo números separados por comas.")
    
    def medidas_tendencia_central(self):
        """Calcula medidas de tendencia central"""
        if self.datos is None:
            print("No hay datos para analizar.")
            return
        
        valores = self.datos['Valores'] if 'Valores' in self.datos.columns else self.datos.iloc[:, 0]
        
        media = np.mean(valores)
        mediana = np.median(valores)
        moda = stats.mode(valores)
        
        print("\n" + "=" * 50)
        print("MEDIDAS DE TENDENCIA CENTRAL")
        print("=" * 50)
        print(f"Media: {media:.4f}")
        print(f"Mediana: {mediana:.4f}")
        print(f"Moda: {moda.mode[0]} (aparece {moda.count[0]} veces)")
    
    def medidas_dispersion(self):
        """Calcula medidas de dispersión"""
        if self.datos is None:
            print("No hay datos para analizar.")
            return
        
        valores = self.datos['Valores'] if 'Valores' in self.datos.columns else self.datos.iloc[:, 0]
        
        varianza = np.var(valores, ddof=1)  # ddof=1 para muestra, ddof=0 para población
        desviacion_estandar = np.std(valores, ddof=1)
        rango = np.max(valores) - np.min(valores)
        rango_intercuartil = np.percentile(valores, 75) - np.percentile(valores, 25)
        
        print("\n" + "=" * 50)
        print("MEDIDAS DE DISPERSIÓN")
        print("=" * 50)
        print(f"Varianza: {varianza:.4f}")
        print(f"Desviación estándar: {desviacion_estandar:.4f}")
        print(f"Rango: {rango:.4f}")
        print(f"Rango intercuartílico: {rango_intercuartil:.4f}")
    
    def analisis_probabilidad(self):
        """Realiza análisis de probabilidad básico"""
        print("\n" + "=" * 50)
        print("ANÁLISIS DE PROBABILIDAD")
        print("=" * 50)
        
        print("1. Probabilidad de un evento")
        print("2. Distribución binomial")
        print("3. Distribución normal")
        print("4. Regresar al menú principal")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            self.probabilidad_evento()
        elif opcion == '2':
            self.distribucion_binomial()
        elif opcion == '3':
            self.distribucion_normal()
        elif opcion == '4':
            return
        else:
            print("Opción no válida.")
    
    def probabilidad_evento(self):
        """Calcula la probabilidad de un evento simple"""
        try:
            casos_favorables = int(input("Número de casos favorables: "))
            casos_totales = int(input("Número de casos totales: "))
            
            if casos_totales == 0:
                print("Error: El número de casos totales no puede ser cero.")
                return
            
            probabilidad = casos_favorables / casos_totales
            print(f"Probabilidad: {probabilidad:.4f} o {probabilidad*100:.2f}%")
        except ValueError:
            print("Error: Ingrese valores numéricos enteros.")
    
    def distribucion_binomial(self):
        """Calcula probabilidades binomiales"""
        try:
            n = int(input("Número de ensayos (n): "))
            p = float(input("Probabilidad de éxito en cada ensayo (p): "))
            k = int(input("Número de éxitos deseados (k): "))
            
            probabilidad = stats.binom.pmf(k, n, p)
            print(f"P(X = {k}) = {probabilidad:.6f}")
            
            # Probabilidad acumulada
            prob_acumulada = stats.binom.cdf(k, n, p)
            print(f"P(X ≤ {k}) = {prob_acumulada:.6f}")
            
        except ValueError:
            print("Error: Ingrese valores numéricos válidos.")
    
    def distribucion_normal(self):
        """Calcula probabilidades para la distribución normal"""
        try:
            media = float(input("Media (μ): "))
            desviacion = float(input("Desviación estándar (σ): "))
            x = float(input("Valor (x): "))
            
            # Calcular probabilidad acumulada
            probabilidad = stats.norm.cdf(x, loc=media, scale=desviacion)
            print(f"P(X ≤ {x}) = {probabilidad:.6f}")
            
            # Calcular valor z
            z = (x - media) / desviacion
            print(f"Valor z: {z:.4f}")
            
        except ValueError:
            print("Error: Ingrese valores numéricos válidos.")
    
    def visualizacion_datos(self):
        """Genera visualizaciones de los datos"""
        if self.datos is None:
            print("No hay datos para visualizar.")
            return
        
        valores = self.datos['Valores'] if 'Valores' in self.datos.columns else self.datos.iloc[:, 0]
        
        print("\n" + "=" * 50)
        print("VISUALIZACIÓN DE DATOS")
        print("=" * 50)
        print("1. Histograma")
        print("2. Diagrama de caja (Boxplot)")
        print("3. Gráfico de dispersión (si hay dos variables)")
        print("4. Regresar al menú principal")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            plt.figure(figsize=(10, 6))
            plt.hist(valores, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Histograma de los datos')
            plt.xlabel('Valores')
            plt.ylabel('Frecuencia')
            plt.grid(axis='y', alpha=0.75)
            plt.show()
        
        elif opcion == '2':
            plt.figure(figsize=(10, 6))
            plt.boxplot(valores)
            plt.title('Diagrama de caja de los datos')
            plt.ylabel('Valores')
            plt.grid(axis='y', alpha=0.75)
            plt.show()
        
        elif opcion == '3':
            if self.datos.shape[1] >= 2:
                x_col = input("Nombre de la columna para el eje X: ")
                y_col = input("Nombre de la columna para el eje Y: ")
                
                if x_col in self.datos.columns and y_col in self.datos.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(self.datos[x_col], self.datos[y_col], alpha=0.7)
                    plt.title('Gráfico de dispersión')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.grid(alpha=0.75)
                    plt.show()
                else:
                    print("Una o ambas columnas no existen en los datos.")
            else:
                print("Se necesitan al menos dos variables para un gráfico de dispersión.")
        
        elif opcion == '4':
            return
        
        else:
            print("Opción no válida.")
    
    def mostrar_menu(self):
        """Muestra el menú principal y maneja las opciones"""
        while True:
            print("\n" + "=" * 50)
            print("MENÚ PRINCIPAL - PROYECTO ESTADÍSTICA")
            print("=" * 50)
            print("1. Medidas de tendencia central")
            print("2. Medidas de dispersión")
            print("3. Análisis de probabilidad")
            print("4. Visualización de datos")
            print("5. Cargar nuevos datos")
            print("6. Salir")
            
            opcion = input("Seleccione una opción: ")
            
            if opcion == '1':
                self.medidas_tendencia_central()
            elif opcion == '2':
                self.medidas_dispersion()
            elif opcion == '3':
                self.analisis_probabilidad()
            elif opcion == '4':
                self.visualizacion_datos()
            elif opcion == '5':
                self.cargar_datos()
            elif opcion == '6':
                print("¡Gracias por usar el Proyecto Python para Análisis Estadístico!")
                break
            else:
                print("Opción no válida. Por favor, seleccione una opción del 1 al 6.")

# Ejecutar el proyecto
if __name__ == "__main__":
    proyecto = ProyectoEstadistica()
    proyecto.mostrar_menu()