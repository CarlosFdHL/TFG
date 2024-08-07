import numpy as np, pandas as pd, matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import subprocess
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import openpyxl
import time
import pysindy as ps
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
from io import StringIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter  
from matplotlib.ticker import FuncFormatter


#############################################################################################################################################################

def ejec_hugoniot():
    subprocess.run(['make'], cwd="/home/carlosfdhl/work/hugoniot", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    resultado = subprocess.run(["/home/carlosfdhl/work/hugoniot/Hugoniot.FEOS"], cwd="/home/carlosfdhl/work/hugoniot", capture_output=True, text=True)

    output = resultado.stdout
    fake_file = StringIO(output)

    df = leer_hugoniot(fake_file)
    return df

#############################################################################################################################################################

def leer_hugoniot(file_input):
    #Para leer archivos de salida del programa hugoniot.F90

    # Función interna para parsear la línea del encabezado y extraer los nombres de las columnas
    def parse_header(header_line):
        header_line = header_line.strip()  # Eliminar espacios al inicio y al final
        header_line = header_line[1:]  # Eliminar el carácter '#' inicial
        column_names = header_line.split(',')  # Separar los nombres de las columnas
        column_names = [name.strip() for name in column_names]  # Limpiar espacios alrededor de los nombres
        return column_names

    # Determinar si file_input es una ruta de archivo o un objeto StringIO
    if isinstance(file_input, StringIO):
        # Es un objeto StringIO, leer directamente de él
        header_line = file_input.readline().strip()  # Leer la primera línea para obtener el encabezado
        column_names = parse_header(header_line)  # Parsear el encabezado para obtener los nombres de las columnas
        data = file_input.readlines()  # Leer el resto de las líneas como datos
        df = pd.read_csv(StringIO(''.join(data)), header=None, sep=r"\s+", names=column_names)
    else:
        # Suponer que file_input es una ruta de archivo, leer desde el sistema de archivos
        with open(file_input, 'r') as file:
            header_line = next(file).strip()  # Leer la primera línea para obtener el encabezado
            column_names = parse_header(header_line)  # Parsear el encabezado para obtener los nombres de las columnas
        # Cargar los datos en un DataFrame
        df = pd.read_csv(file_input, skiprows=1, header=None, sep=r"\s+", names=column_names)

    # Transformaciones de las columnas según lo solicitado previamente
    df = df.rename(columns={'r(i)': 'R (kg/m3)', 't(i)': 'T (K)', 'p(i)': 'P (GPa)', 'e(i)':'e (MJ/kg)'})
    df['R (kg/m3)'] = df['R (kg/m3)'] * 1000
    df['R/R0'] = df['R (kg/m3)'] / df['R (kg/m3)'].iloc[0]
    df['T (K)'] = df['T (K)'] * 11604
    df['P (GPa)'] = df['P (GPa)'] * 10**(-9)
    df['e (MJ/kg)'] = df['e (MJ/kg)'] * 10**(-10) #erg/g a MJ/kg


    return df

#############################################################################################################################################################

def leer_archivo_hugoniot(path): #Lee el archivo en 'path' y devuelve el Data Frame 'datos_Al_hug' habiendo cambiado las unidades
    try:
        datos_Al_hug = pd.read_csv(path, sep='  ')

        for columna in datos_Al_hug.columns:
            datos_Al_hug[columna] = pd.to_numeric(datos_Al_hug[columna], errors='coerce')

        #Elimino la última fila del Data Frame: 
        datos_Al_hug = datos_Al_hug.iloc[:-1, :]

        #Cambio las unidades:
        segunda_columna = datos_Al_hug.iloc[:, 1].tolist()
        segunda_columna = [x * 11606 for x in segunda_columna] #1 eV = 11606 K
        datos_Al_hug.iloc[:, 1] = segunda_columna

        tercera_columna = datos_Al_hug.iloc[:, 2].tolist()
        tercera_columna = [x * 100 for x in tercera_columna] #10e+12 dyne/cm^2 = 100 GPa
        datos_Al_hug.iloc[:, 2] = tercera_columna

        cuarta_columna = datos_Al_hug.iloc[:, 3].tolist()
        cuarta_columna = [x * 10**(-7) for x in cuarta_columna] #1 erg/g = 10e-7 kJ/kg
        datos_Al_hug.iloc[:, 3] = cuarta_columna

        quinta_columna = datos_Al_hug.iloc[:, 4].tolist()
        quinta_columna = [x * 1 for x in quinta_columna] #10e+5 cm/s = 1 km/s 
        datos_Al_hug.iloc[:, 4] = quinta_columna

        sexta_columna = datos_Al_hug.iloc[:, 5].tolist()
        sexta_columna = [x * 1 for x in sexta_columna] #10e+5 cm/s = 1 km/s 
        datos_Al_hug.iloc[:, 5] = sexta_columna

        #Renombro las columnas: 
        nuevos_nombres = ["R (g/cm3)","T (K)", "P (GPa)", "E (kJ/kg)", "Us (km/s)", "Up (km/s)"]
        nuevos_nombres = dict(zip(datos_Al_hug.columns, nuevos_nombres))

        datos_Al_hug.rename(columns= nuevos_nombres, inplace=True)

        #Añado la columna R/R0
        datos_Al_hug['R/R0'] = datos_Al_hug.iloc[:, 0] / datos_Al_hug.iloc[0, 0]

        #Imprimo el df 
        #print(datos_Al_hug)
        return datos_Al_hug

    except FileNotFoundError:
        print(f"El archivo '{path}' no fue encontrado.")

#############################################################################################################################################################

def procesar_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
    
    lineas = lineas[1:] #Excluir la primera línea
    

    primera_linea = np.array(lineas[0].strip().split('  '), dtype=float)
    segunda_linea = np.array(lineas[1].strip().split('  '), dtype=float)
    
    dataframe_pandas = pd.DataFrame(linea.strip().split('  ') for linea in lineas[2:]).T

    return primera_linea, segunda_linea, dataframe_pandas

#############################################################################################################################################################

def ejec_showeos(elemento, operacion): #ejecutar el programa showeos para un elemento (ej:Al) y un modo de operación: 1- Isoterma, 2- Isócoras, 3-Isentrópicas, 4-Mountain-Plot, 5- Hugoniot, 6- Info. pto individual
    subprocess.call(["/home/carlosfdhl/work/FEOS/run/showeos", elemento, operacion], cwd = "/home/carlosfdhl/work/FEOS/run" , stdout=subprocess.DEVNULL, stderr = subprocess.DEVNULL)

#############################################################################################################################################################

def ejec_feos(elemento):
    subprocess.call(["/home/carlosfdhl/work/FEOS/run/feos", elemento], cwd = "/home/carlosfdhl/work/FEOS/run",stdout=subprocess.DEVNULL,  stderr = subprocess.DEVNULL )

#############################################################################################################################################################

def filtrar_dataframe(df, columna, valor_referencia, porcentaje):
    limite_inferior = valor_referencia * (1 - porcentaje)
    limite_superior = valor_referencia * (1 + porcentaje)
    df_filtrado = df[(df[columna] > limite_inferior) & (df[columna] < limite_superior)]
    return df_filtrado

#############################################################################################################################################################

def editar_palabra_en_archivo(path, inicio_linea, palabra_a_editar, nueva_palabra):
    try:
        with open(path, 'r') as file:
            lineas = file.readlines()
            

        encontrado = False

        with open(path, 'w') as file:
            for linea in lineas:
                if linea.startswith(inicio_linea) and not encontrado:
                    palabras = linea.split()
                    for i in range(len(palabras)):
                        if palabras[i] == palabra_a_editar:
                            palabras[i] = str(nueva_palabra)
                            encontrado = True
                            break
                    nueva_linea = '  '.join(palabras)
                    file.write(nueva_linea + '\n')
                else:
                    file.write(linea)
        
        if not encontrado:
            print(f"No se encontró ninguna línea que comience con '{inicio_linea}' o la palabra '{palabra_a_editar}' no está presente.")
        else:
            print(f"La palabra '{palabra_a_editar}' en la línea que comienza con '{inicio_linea}' ha sido reemplazada por '{nueva_palabra}'.")
        
    except FileNotFoundError:
        print(f"El archivo '{path}' no fue encontrado.")

#############################################################################################################################################################

def editar_fichero(path, inicio_linea, nueva_palabra):
    try:
        with open(path,'r') as file:
            lineas = file.readlines()
            lineas_originales = lineas
        encontrado = False
        with open(path, 'w') as file:
            for linea in lineas:
                if inicio_linea in linea:
                    palabras = linea.split()
                    if palabras[0]== str(inicio_linea) and palabras[1] == "=":
                        encontrado = True
                        espacios_iniciales = linea[:len(linea) - len(linea.lstrip())]
                        linea = espacios_iniciales + palabras[0] + " "+ palabras[1]+ " " + str(nueva_palabra) + "\n" 
                        print(f"La línea que comienza con '{inicio_linea}' ha sido reemplazada por '{nueva_palabra}'.")
                file.write(linea)
        if not encontrado:
            print(f"No se encontró ninguna línea que comience con '{inicio_linea}'")
        file.close()
    except Exception as e:
        print("Ocurrió un error:", e)
        # Restaurar el archivo original
        try:
            with open(path, 'w') as file:
                file.writelines(lineas_originales)
                print(f"El archivo original ha sido restaurado a partir de la copia de seguridad: {path}")
        except Exception as e:
            print("No se pudo restaurar el archivo original:", e)

#############################################################################################################################################################

def editar_palabra_en_archivo(path, inicio_linea, palabra_a_editar, nueva_palabra):
    try:
        with open(path, 'r') as file:
            lineas = file.readlines()

        encontrado = False

        with open(path, 'w') as file:
            for linea in lineas:
                # Verifica si inicio_linea está en la línea, en lugar de verificar si la línea comienza con inicio_linea
                if inicio_linea in linea and not encontrado:
                    # Intenta reemplazar directamente la palabra_a_editar por nueva_palabra en la línea
                    nueva_linea = linea.replace(palabra_a_editar, str(nueva_palabra))
                    if nueva_linea != linea:  # Si se realizó un reemplazo
                        encontrado = True
                        linea = nueva_linea
                file.write(linea)
        
        if not encontrado:
            print(f"No se encontró ninguna línea que contenga '{inicio_linea}' o la palabra '{palabra_a_editar}' no está presente.")
        else:
            print(f"La palabra '{palabra_a_editar}' en la línea que contiene '{inicio_linea}' ha sido reemplazada por '{nueva_palabra}'.")
        
    except FileNotFoundError:
        print(f"El archivo '{path}' no fue encontrado.")

#############################################################################################################################################################

def custom_scientific_format(num):
    # Convertir el número a notación científica estándar
    num_scientific = "{:.6e}".format(num)
    # Dividir la parte decimal y el exponente
    parts = num_scientific.split('e')
    # Ajustar el exponente para eliminar el cero innecesario
    exponent = parts[1]
    exponent_sign = exponent[0]
    if exponent[1] == '0':  # Si el exponente tiene un 0 después del signo, quitarlo
        exponent = exponent_sign + exponent[2:]
    # Reconstruir la notación científica
    custom_format = parts[0] + 'e' + exponent
    return custom_format

#############################################################################################################################################################

def encontrar_valores_equiespaciados(x1, n_puntos_salida):
    valor_inicial = x1[0]
    valor_maximo = max(x1)
    
    # Verificar si el número de puntos solicitados es mayor que la longitud de x1
    if n_puntos_salida > len(x1):
        raise ValueError("El número de puntos solicitados excede el número de puntos disponibles en x1.")
    
    # Lista para almacenar los índices aproximados de los puntos equiespaciados
    indices_equiespaciados = np.linspace(0, x1.index(valor_maximo) - 1, n_puntos_salida, dtype=int)
    
    # Seleccionar los valores de x1 correspondientes a los índices calculados
    valores_equiespaciados = [x1[i] for i in indices_equiespaciados]
    
    return valores_equiespaciados

#############################################################################################################################################################

def eliminar_nan_2VectoresDependientes(vect_x, vect_y):
    indices_a_eliminar = np.isnan(vect_x) | np.isnan(vect_y)
    
    vect_x_sin_nan = vect_x[~indices_a_eliminar]
    vect_y_sin_nan = vect_y[~indices_a_eliminar]
    
    return vect_x_sin_nan, vect_y_sin_nan

#############################################################################################################################################################

def annadir_ndatos_feos(x, y, k):
    matriz_datos = np.column_stack((x,y))
    matriz_datos_nuevos=[]
    for i in range(len(x)-1):
        m = matriz_datos[i+1, 1] - matriz_datos[i, 1]
        n = matriz_datos[i,1] - m * x[i]
        x_nuevo = np.linspace(x[i], x[i+1], k+2)[1:-1] #Añadir 5 valores entre cada dos puntos (x,y)
        y_nuevo = m*x_nuevo+n
        for a, b in zip(x_nuevo, y_nuevo):
            matriz_datos_nuevos.append([a,b])
    datos = np.vstack((matriz_datos, matriz_datos_nuevos))
    datos = datos[datos[:, 1].argsort()]

    return datos

#############################################################################################################################################################

def eliminar_nan_vector(vect_x):
    indices_a_eliminar = np.isnan(vect_x) 
    vect_x_sin_nan = vect_x[~indices_a_eliminar]

    return vect_x_sin_nan

#############################################################################################################################################################

def convert_list_to_float_array(data_list):
    """
    Convierte una lista de cadenas a un array de NumPy de tipo float.
    Las cadenas que no se pueden convertir a float, incluidas las cadenas vacías y los espacios, se reemplazan por np.nan.
    
    :param data_list: Lista de cadenas a convertir.
    :return: Array de NumPy de los datos convertidos a tipo float.
    """
    sanitized = []
    for item in data_list:
        if item.strip() == "":
            # Añadir np.nan si la cadena está vacía o solo contiene espacios en blanco
            sanitized.append(np.nan)
        else:
            try:
                # Intentar convertir la cadena a float
                sanitized.append(item)
            except ValueError:
                # Añadir np.nan si la conversión falla
                sanitized.append(np.nan)
    return np.array(sanitized, dtype=float)

#############################################################################################################################################################

def leer_fpeos():
    # Inicializar una lista para almacenar los diccionarios de cada línea
    data = []

    # Abrir y leer el archivo línea por línea
    with open('/home/carlosfdhl/work/FPEOS/FPEOS_Hugoniot.txt', 'r') as file:
        for line in file:
            # Dividir la línea por espacios y filtrar los elementos vacíos
            parts = [part for part in line.split(' ') if part]
            
            # Inicializar un diccionario para almacenar los pares clave-valor de esta línea
            line_data = {}
            
            # Recorrer cada parte para extraer las claves y valores
            key = None
            for part in parts:
                # Si encontramos un "=", establecemos la clave y esperamos el valor en la siguiente iteración
                if "=" in part:
                    key = part.replace('=', '').strip()
                elif key:
                    # Almacenamos el valor asociado a la clave y reiniciamos la clave para el próximo par
                    line_data[key] = float(part.replace('=', ''))
                    key = None
            
            # Añadir el diccionario de la línea actual a la lista de datos
            data.append(line_data)

    # Convertir la lista de diccionarios en un DataFrame
    df = pd.DataFrame(data)
    nuevos_nombres = ["T (K)","R/R0", "R0 (kg/m3)", "R (kg/m3)", "P (GPa)", "E (Ha)", "E0 (Ha)", "(E-E0) (Ha)", "Hug (Ha)", "Up (km/s)", "Us (km/s)"]
    nuevos_nombres = dict(zip(df.columns, nuevos_nombres))
    
    df.rename(columns= nuevos_nombres, inplace=True)
    df['R (kg/m3)'] = df['R (kg/m3)'] * 1000
    df['R0 (kg/m3)'] = df['R0 (kg/m3)'] * 1000
    return df

#############################################################################################################################################################

def CambiarAPoliestireno():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '3717', '1000' ) 
    editar_palabra_en_archivo(filename, 'r0=', '2.7', '1.05' )  

#############################################################################################################################################################

def CambiarDePoliestirenoAAluminio():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '1000', '3717' ) 
    editar_palabra_en_archivo(filename, 'r0=', '1.05', '2.7' ) 

#############################################################################################################################################################

def CambiarAOro():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '3717', '2700' ) 
    editar_palabra_en_archivo(filename, 'r0=', '2.7', '19.3' )  

#############################################################################################################################################################

def CambiarDeOroAAluminio():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '2700', '3717' ) 
    editar_palabra_en_archivo(filename, 'r0=', '19.3', '2.7' ) 

#############################################################################################################################################################

def CambiarADT():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '3717', '1001' ) 
    editar_palabra_en_archivo(filename, 'r0=', '2.7', '0.2' )  

#############################################################################################################################################################

def CambiarDeDTAAluminio():
    filename = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(filename, 'ed%eosnumber=', '1001', '3717' ) 
    editar_palabra_en_archivo(filename, 'r0=', '0.2', '2.7' ) 

#############################################################################################################################################################


def concat_RP(r_feos, p_feos, r_fpeos, p_fpeos):
    """Concatena los vectores y los ordena de 
    menor a mayor en función de la presión"""

    p = np.concatenate((p_feos, p_fpeos))
    r = np.concatenate((r_feos, r_fpeos))

    
    indices_ordenados = np.argsort(p)

    p = p[indices_ordenados]
    r = r[indices_ordenados]
    
    return r,p

#############################################################################################################################################################

def config_logPlots(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    #GRID
    ax.grid()
    ax.xaxis.set_major_locator(ticker.AutoLocator()) 
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1)) 
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)

#############################################################################################################################################################

def config_linPlots(ax):
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='-', linewidth=1)

#############################################################################################################################################################

def buscar_1parametro(path_archivo_entrada, inicio_linea, valor_actual, r_exp, p_exp):
    #Asegurar que el valor inicial de la búsqueda es: valor_inicial
    valor_inicial = valor_actual
    valor_actual = float(valor_actual)
    probar = [valor_actual*0.7, valor_actual*0.8, valor_actual*0.9, valor_actual, valor_actual*1.1, valor_actual*1.2, valor_actual*1.3]
    valor_actual = "{:g}".format(valor_actual)
    valor_actual = valor_actual.replace("+", "")

    error_cuadratico_medio = []
    df = []

    for valor in probar:

        valor = "{:g}".format(valor)
        valor = valor.replace("+", "")
        editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor)
        valor_actual = valor

        datos_feos = ejec_hugoniot()
        p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
        r_feos = datos_feos["R (kg/m3)"].to_numpy(dtype=float)
        
        #df.append(datos_teoricos)
        interpolacion, error = interp_datos_experimentales(p_feos,r_feos,p_exp, r_exp)
        print('ERROR = ', error)
        error_cuadratico_medio.append(error)
    
    izda = sum(error_cuadratico_medio[:3])
    dcha = sum(error_cuadratico_medio[-3:])
    error_actual = error_cuadratico_medio[3]

    valor= float(valor_actual)/1.3
    valor = "{:g}".format(valor)
    valor = valor.replace("+", "")
    editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor)
    valor_actual = valor

    if izda<dcha:
        encontrado = False
        error_actual = error_cuadratico_medio[3]
        while not encontrado:
            try:

                valor = float(valor_actual)*0.99
                valor = "{:g}".format(valor)
                valor = valor.replace("+", "")
                editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor)
                valor_anterior = valor
                valor_actual = valor

                datos_feos = ejec_hugoniot()
                p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
                r_feos = datos_feos["R (kg/m3)"].to_numpy(dtype=float)


                interpolacion, error = interp_datos_experimentales(p_feos, r_feos, p_exp, r_exp)
                if int(error*1) >= int(error_actual*1):
                    print("ENCONTRADO")
                    encontrado = True
                    editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor_anterior)
                    return valor_anterior
                else:
                    error_actual = error
                    print("ERROR ACTUAL =", error_actual)
            except: pass

    elif izda>dcha:
        encontrado = False
        error_actual = error_cuadratico_medio[3]
        while not encontrado:
            try:
                valor = float(valor_actual)*1.01
                valor = "{:g}".format(valor)
                valor = valor.replace("+", "")
                editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor)
                valor_anterior = valor
                valor_actual = valor

                datos_feos = ejec_hugoniot()
                p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
                r_feos = datos_feos["R (kg/m3)"].to_numpy(dtype=float)


                interpolacion, error = interp_datos_experimentales(p_feos, r_feos, p_exp, r_exp)

                if int(error*1000) >= int(error_actual*1000):
                    print("ENCONTRADO")
                    encontrado = True
                    editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor_anterior)
                    return valor_anterior
                else:
                    error_actual = error
                    print("ERROR ACTUAL =", error_actual)
            except: pass
    else:
        print("El parámetro no afecta a la calidad de la aproximación")
        editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor_inicial)

        return valor_inicial
    
    editar_palabra_en_archivo(path_archivo_entrada, inicio_linea, valor_actual, valor)

#############################################################################################################################################################

def interp_datos_experimentales(x_conocidos, y_conocidos, x_aproximar, y_aproximar):
    max_x_conocido = max(x_conocidos)
    min_x_conocido = min(x_conocidos)
    
    max_x_conocido_index = np.where(x_conocidos == max_x_conocido)[0]
    matriz_datos_conocidos = []
    
    for x, y in zip(x_conocidos, y_conocidos):
        row = np.array([x, y])
        matriz_datos_conocidos.append(row.tolist())
    matriz_datos_conocidos = np.array(matriz_datos_conocidos, dtype = float)

    
    interpolacion = []
    error_cuadratico_medio = 0

    for valor, y_aprox in zip(x_aproximar, y_aproximar):
        
        if valor >= min_x_conocido and valor < max_x_conocido:
            
            idx_menor = np.argmin(np.abs(matriz_datos_conocidos[:, 0] - valor ))
            idx_mayor = idx_menor + 1
            x_menor, y_menor = matriz_datos_conocidos[idx_menor]
            if x_menor == max_x_conocido:
                idx_menor -=1
                idx_mayor -=1
                x_menor, y_menor = matriz_datos_conocidos[idx_menor]
            if x_menor < max_x_conocido:
                x_mayor, y_mayor = matriz_datos_conocidos[idx_mayor]
                if y_menor < y_mayor:
                    encontrado = False
                    while (encontrado == False):
                        if x_menor<=valor:
                            if x_mayor >= valor:
                                encontrado = True
                            else: 
                                idx_mayor += 1
                                x_mayor, y_mayor = matriz_datos_conocidos[idx_mayor]
                        else:
                            idx_menor -= 1
                            x_menor , y_menor = matriz_datos_conocidos[idx_menor]
                            idx_mayor -= 1
                            x_mayor, y_mayor = matriz_datos_conocidos[idx_mayor]
                    m = (y_mayor - y_menor) / (x_mayor - x_menor)
                    n = y_menor - m * x_menor
                    y_interpolado = m*valor+n
                    
                else: 
                    encontrado = False
                    while (encontrado == False):
                        if x_mayor <=valor:
                            if x_menor>=valor:
                                encontrado = True
                            else:
                                idx_menor+=1
                                x_menor, y_menor = matriz_datos_conocidos[idx_menor]
                        else:
                            idx_mayor-=1
                            x_mayor, y_mayor = matriz_datos_conocidos[idx_mayor]
                            idx_menor -= 1
                            x_menor , y_menor = matriz_datos_conocidos[idx_menor]
                    m = (y_menor - y_mayor) / (x_menor - x_mayor)
                    n = y_menor - m * x_menor
                    y_interpolado = m*valor+n
                
                error_cuadratico_medio += (y_interpolado-y_aprox)**2
                row = np.array(y_interpolado)
                interpolacion.append(row.tolist())
            
        else :
            y = y_conocidos[int(len(y_conocidos))-1]
            error_cuadratico_medio += (y-y_aprox)**2
            interpolacion.append(y)
    interpolacion = np.array(interpolacion)
    error = error_cuadratico_medio/len(interpolacion)

    return interpolacion, error

#############################################################################################################################################################

def CambiarBulk(linea, valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/FEOS/EOS-Data/FEOS_Material-DB.dat"
    editar_palabra_en_archivo(path_programa, linea, valor_actual, valor_nuevo)
    valor_actual = valor_nuevo

#############################################################################################################################################################

def CambiarAlpha(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'alpha = ', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo
    
#############################################################################################################################################################

def CambiarBeta(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'beta = ', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo
    return valor_actual

#############################################################################################################################################################

def CambiarT00(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'T00 = ', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo
    return valor_actual

#############################################################################################################################################################

def CambiarR00(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'R00 = ', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo
    return valor_actual

#############################################################################################################################################################

def CambiarT01(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'T01 = ', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo
    return valor_actual

#############################################################################################################################################################

def CambiarMultiplicador(valor_actual, valor_nuevo):
    path_programa = "/home/carlosfdhl/work/hugoniot/hugoniot.F90"
    editar_palabra_en_archivo(path_programa, 'TipoMultiplicador =', valor_actual, valor_nuevo)
    valor_actual = valor_nuevo    

#############################################################################################################################################################

def errorMultiplicador(r_exp,p_exp):
    """Ejecuta y lee el programa Hugoniot
    Obtiene los datos y los compara con datos experimentales
    Finalmente calcula el error"""

    datos_feos = ejec_hugoniot()
    p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
    r_feos = datos_feos["R (kg/m3)"].to_numpy(dtype=float)

    interp, error = interp_datos_experimentales(p_feos, r_feos, p_exp, r_exp)
    error_minR = ((min(r_feos)) - (min(r_exp))) ** 2
    error_maxR = ((max(r_feos)) - (max(r_exp))) ** 2
    error_maxP = (r_feos[np.argmax(p_feos)]-r_exp[np.argmax(p_exp)]) ** 2

    error =error + error_minR +  error_maxR +  error_maxP

    return error

#############################################################################################################################################################

def optimizarParametrosMultiplicador(r_exp, p_exp, alphaActual, delta_Alpha, betaActual, delta_Beta, t00Actual, delta_T00, r00Actual, delta_R00):
    error_anterior = errorMultiplicador(r_exp, p_exp)
    encontrado = False
    while not encontrado:

        CambiarAlpha(alphaActual, str(abs(float(alphaActual) + delta_Alpha)))

        error_alpha = errorMultiplicador(r_exp, p_exp)
        error_alpha = [error_alpha, 'alpha']

        CambiarAlpha(str(abs(float(alphaActual) + delta_Alpha)), alphaActual)

        CambiarBeta(betaActual, str(abs(float(betaActual) + delta_Beta)))
        
        error_beta = errorMultiplicador(r_exp, p_exp)
        error_beta = [error_beta, 'beta']

        CambiarBeta(str(abs(float(betaActual) + delta_Beta)), betaActual)

        CambiarT00(t00Actual, str(abs(float(t00Actual) + delta_T00)))

        error_t00 = errorMultiplicador(r_exp, p_exp)
        error_t00 = [error_t00, 't00']
        
        CambiarT00(str(abs(float(t00Actual) + delta_T00)), t00Actual)

        CambiarR00(r00Actual, str(abs(float(r00Actual) + delta_R00)))

        error_r00 = errorMultiplicador(r_exp, p_exp)
        error_r00 = [error_r00, 'r00']

        CambiarR00(str(abs(float(r00Actual) + delta_R00)), r00Actual)


        errores = [error_alpha, error_beta, error_t00, error_r00]
        menor_error = min(errores, key=lambda x: x[0])

        if menor_error[0] <= error_anterior:
            if menor_error[1] == 'alpha':
                if float(alphaActual)+delta_Alpha <0:
                    CambiarAlpha(alphaActual, '0.001')
                    alphaActual = '0.001'
                    delta_Alpha = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarAlpha(alphaActual, str(float(alphaActual) + delta_Alpha))
                    alphaActual = str(float(alphaActual)+delta_Alpha)
            
            elif menor_error[1] == 'beta':
                if float(betaActual)+delta_Beta <0:
                    CambiarBeta(betaActual, '0.001')
                    betaActual = '0.001'
                    delta_Beta = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBeta(betaActual, str(float(betaActual)+delta_Beta))
                    betaActual = str(float(betaActual)+delta_Beta)
            
            elif menor_error[1] == 't00':
                if float(t00Actual)+delta_T00 <0:
                    CambiarT00(t00Actual, '0.001')
                    t00Actual = '0.001'
                    delta_T00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarT00(t00Actual, str(float(t00Actual)+delta_T00))
                    t00Actual = str(float(t00Actual)+delta_T00)
            
            elif menor_error[1] == 'r00':
                if float(r00Actual)+delta_R00 <0:
                    CambiarR00(r00Actual, '0.001')
                    r00Actual = '0.001'
                    delta_R00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarR00(r00Actual, str(float(r00Actual)+delta_R00))
                    r00Actual = str(float(r00Actual)+delta_R00)


        if error_alpha[0] >= error_anterior:
            delta_Alpha = -0.97 * delta_Alpha
        if error_beta[0] >= error_anterior:
            delta_Beta = -0.97 * delta_Beta
        if error_t00[0] >= error_anterior:
            delta_T00 = -0.97 * delta_T00
        if error_r00[0] >= error_anterior:
            delta_R00 = -0.97 * delta_R00

        if menor_error[0]< error_anterior:
            error_anterior = menor_error[0]
        print("ERROR ANTERIOR = ", error_anterior)


        if abs(delta_Alpha/float(alphaActual)) < 10**(-4) and abs(delta_Beta/float(betaActual)) < 10**(-4) and abs(delta_T00/float(t00Actual)) < 10**(-4) and abs(delta_R00/float(r00Actual)) < 10**(-4):
            print("alpha = ", alphaActual, "delta = ", delta_Alpha)
            print("beta = ", betaActual, "delta = ", delta_Beta)
            print("t00 = ", t00Actual, "delta = ", delta_T00)
            print("r00 = ", r00Actual, "delta = ", delta_R00)
            encontrado = True 
    
    return alphaActual, betaActual, t00Actual, r00Actual

#############################################################################################################################################################

def optimizarParametrosMultiplicador_B(r_exp, p_exp, BActual, delta_B, alphaActual, delta_Alpha, betaActual, delta_Beta, t00Actual, delta_T00, r00Actual, delta_R00):
    error_anterior = errorMultiplicador(r_exp, p_exp)
    encontrado = False
    while not encontrado:

        CambiarBulk('[2700]_Bulk-Modulus =', BActual, str(abs(float(BActual)+ delta_B)))

        error_B = errorMultiplicador(r_exp, p_exp)
        error_B = [error_B, 'bulk']

        CambiarBulk('[2700]_Bulk-Modulus =', str(abs(float(BActual)+ delta_B)), BActual)

        CambiarAlpha(alphaActual, str(abs(float(alphaActual) + delta_Alpha)))

        error_alpha = errorMultiplicador(r_exp, p_exp)
        error_alpha = [error_alpha, 'alpha']

        CambiarAlpha(str(abs(float(alphaActual) + delta_Alpha)), alphaActual)

        CambiarBeta(betaActual, str(abs(float(betaActual) + delta_Beta)))
        
        error_beta = errorMultiplicador(r_exp, p_exp)
        error_beta = [error_beta, 'beta']

        CambiarBeta(str(abs(float(betaActual) + delta_Beta)), betaActual)

        CambiarT00(t00Actual, str(abs(float(t00Actual) + delta_T00)))

        error_t00 = errorMultiplicador(r_exp, p_exp)
        error_t00 = [error_t00, 't00']
        
        CambiarT00(str(abs(float(t00Actual) + delta_T00)), t00Actual)

        CambiarR00(r00Actual, str(abs(float(r00Actual) + delta_R00)))

        error_r00 = errorMultiplicador(r_exp, p_exp)
        error_r00 = [error_r00, 'r00']

        CambiarR00(str(abs(float(r00Actual) + delta_R00)), r00Actual)


        errores = [error_alpha, error_beta, error_t00, error_r00]
        menor_error = min(errores, key=lambda x: x[0])

        if menor_error[0] < error_anterior:
            if menor_error[1] == 'bulk':
                if float(BActual)+delta_B <0:
                    CambiarBulk(BActual, '0.001')
                    BActual = '0.001'
                    delta_B = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBulk(BActual, str(float(BActual) + delta_B))
                    BActual = str(float(BActual)+delta_B)
            elif menor_error[1] == 'alpha':
                if float(alphaActual)+delta_Alpha <0:
                    CambiarAlpha(alphaActual, '0.001')
                    alphaActual = '0.001'
                    delta_Alpha = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarAlpha(alphaActual, str(float(alphaActual) + delta_Alpha))
                    alphaActual = str(float(alphaActual)+delta_Alpha)
            
            elif menor_error[1] == 'beta':
                if float(betaActual)+delta_Beta <0:
                    CambiarBeta(betaActual, '0.001')
                    betaActual = '0.001'
                    delta_Beta = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBeta(betaActual, str(float(betaActual)+delta_Beta))
                    betaActual = str(float(betaActual)+delta_Beta)
            
            elif menor_error[1] == 't00':
                if float(t00Actual)+delta_T00 <0:
                    CambiarT00(t00Actual, '0.001')
                    t00Actual = '0.001'
                    delta_T00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarT00(t00Actual, str(float(t00Actual)+delta_T00))
                    t00Actual = str(float(t00Actual)+delta_T00)
            
            elif menor_error[1] == 'r00':
                if float(r00Actual)+delta_R00 <0:
                    CambiarR00(r00Actual, '0.001')
                    r00Actual = '0.001'
                    delta_R00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarR00(r00Actual, str(float(r00Actual)+delta_R00))
                    r00Actual = str(float(r00Actual)+delta_R00)

        if error_B[0] >= error_anterior:
            delta_B = -0.97 * delta_B
        if error_alpha[0] >= error_anterior:
            delta_Alpha = -0.97 * delta_Alpha
        if error_beta[0] >= error_anterior:
            delta_Beta = -0.97 * delta_Beta
        if error_t00[0] >= error_anterior:
            delta_T00 = -0.97 * delta_T00
        if error_r00[0] >= error_anterior:
            delta_R00 = -0.97 * delta_R00

        if menor_error[0]< error_anterior:
            error_anterior = menor_error[0]
        print("ERROR ANTERIOR = ", error_anterior)


        if abs(delta_B/float(BActual)) < 10**(-4) and abs(delta_Alpha/float(alphaActual)) < 10**(-4) and abs(delta_Beta/float(betaActual)) < 10**(-4) and abs(delta_T00/float(t00Actual)) < 10**(-4) and abs(delta_R00/float(r00Actual)) < 10**(-4):
            print("Bulk = ", BActual, "delta = ", delta_B)
            print("alpha = ", alphaActual, "delta = ", delta_Alpha)
            print("beta = ", betaActual, "delta = ", delta_Beta)
            print("t00 = ", t00Actual, "delta = ", delta_T00)
            print("r00 = ", r00Actual, "delta = ", delta_R00)
            encontrado = True 
    
    return BActual, alphaActual, betaActual, t00Actual, r00Actual

#############################################################################################################################################################

def optimizarParametrosMultiplicador_Tipo4(r_exp, p_exp, BActual, delta_B, alphaActual, delta_Alpha, betaActual, delta_Beta, t00Actual, delta_T00, r00Actual, delta_R00, t01Actual, delta_T01):
    error_anterior = errorMultiplicador(r_exp, p_exp)
    encontrado = False
    while not encontrado:

        CambiarBulk('[1000]_Bulk-Modulus =', BActual, str(abs(float(BActual)+ delta_B)))

        error_B = errorMultiplicador(r_exp, p_exp)
        error_B = [error_B, 'bulk']

        CambiarBulk('[1000]_Bulk-Modulus =', str(abs(float(BActual)+ delta_B)), BActual)

        CambiarAlpha(alphaActual, str(abs(float(alphaActual) + delta_Alpha)))

        error_alpha = errorMultiplicador(r_exp, p_exp)
        error_alpha = [error_alpha, 'alpha']

        CambiarAlpha(str(abs(float(alphaActual) + delta_Alpha)), alphaActual)

        CambiarBeta(betaActual, str(abs(float(betaActual) + delta_Beta)))
        
        error_beta = errorMultiplicador(r_exp, p_exp)
        error_beta = [error_beta, 'beta']

        CambiarBeta(str(abs(float(betaActual) + delta_Beta)), betaActual)

        CambiarT00(t00Actual, str(abs(float(t00Actual) + delta_T00)))

        error_t00 = errorMultiplicador(r_exp, p_exp)
        error_t00 = [error_t00, 't00']
        
        CambiarT00(str(abs(float(t00Actual) + delta_T00)), t00Actual)

        CambiarR00(r00Actual, str(abs(float(r00Actual) + delta_R00)))

        error_r00 = errorMultiplicador(r_exp, p_exp)
        error_r00 = [error_r00, 'r00']

        CambiarR00(str(abs(float(r00Actual) + delta_R00)), r00Actual)

        CambiarT01(t01Actual, str(abs(float(t01Actual) + delta_T01)))

        try:
            error_t01 = errorMultiplicador(r_exp, p_exp)
            error_t01 = [error_t01, 't01']
        except Exception as e:
            error_t01 = 999999999999999
            error_t01 = [error_t01, 't01']
            print("**** Error : ", e)

        CambiarT01(str(abs(float(t01Actual) + delta_T01)), t01Actual)

        errores = [error_B, error_alpha, error_beta, error_t00, error_r00, error_t01]
        menor_error = min(errores, key=lambda x: x[0])

        if menor_error[0] < error_anterior:
            if menor_error[1] == 'bulk':
                if float(BActual)+delta_B <0:
                    delta_B = 0.05*delta_B
                    #CambiarBulk('[1000]_Bulk-Modulus =',BActual, '0.001')
                    #BActual = '0.001'
                    #delta_B = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBulk('[1000]_Bulk-Modulus =', BActual, str(float(BActual) + delta_B))
                    BActual = str(float(BActual)+delta_B)
            elif menor_error[1] == 'alpha':
                if float(alphaActual)+delta_Alpha <0:
                    CambiarAlpha(alphaActual, '0.001')
                    alphaActual = '0.001'
                    delta_Alpha = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarAlpha(alphaActual, str(float(alphaActual) + delta_Alpha))
                    alphaActual = str(float(alphaActual)+delta_Alpha)
            
            elif menor_error[1] == 'beta':
                if float(betaActual)+delta_Beta <0:
                    CambiarBeta(betaActual, '0.001')
                    betaActual = '0.001'
                    delta_Beta = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBeta(betaActual, str(float(betaActual)+delta_Beta))
                    betaActual = str(float(betaActual)+delta_Beta)
            
            elif menor_error[1] == 't00':
                if float(t00Actual)+delta_T00 <0:
                    CambiarT00(t00Actual, '0.001')
                    t00Actual = '0.001'
                    delta_T00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarT00(t00Actual, str(float(t00Actual)+delta_T00))
                    t00Actual = str(float(t00Actual)+delta_T00)
            
            elif menor_error[1] == 'r00':
                if float(r00Actual)+delta_R00 <0:
                    CambiarR00(r00Actual, '0.001')
                    r00Actual = '0.001'
                    delta_R00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarR00(r00Actual, str(float(r00Actual)+delta_R00))
                    r00Actual = str(float(r00Actual)+delta_R00)

            elif menor_error[1] == 't01':
                if float(t01Actual)+delta_T01 <1:
                    CambiarT01(r00Actual, '1')
                    t01Actual = '1'
                    delta_T01 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarT01(t01Actual, str(float(t01Actual)+delta_T01))
                    t01Actual = str(float(t01Actual)+delta_T01)

        if error_B[0] > error_anterior:
            delta_B = -0.9 * delta_B
        if error_alpha[0] >= error_anterior:
            delta_Alpha = -0.97 * delta_Alpha
        if error_beta[0] >= error_anterior:
            delta_Beta = -0.97 * delta_Beta
        if error_t00[0] >= error_anterior:
            delta_T00 = -0.97 * delta_T00
        if error_r00[0] >= error_anterior:
            delta_R00 = -0.97 * delta_R00
        if error_t01[0] >= error_anterior:
            delta_T01 = -0.97 * delta_T01
        if error_t01[0] == 999999999999999 : 
            delta_T01 = -0.8 * delta_T01

        if menor_error[0]< error_anterior:
            error_anterior = menor_error[0]
        print("ERROR ANTERIOR = ", error_anterior)


        if abs(delta_B/float(BActual)) < 10**(-4) and abs(delta_Alpha/float(alphaActual)) < 10**(-4) and abs(delta_Beta/float(betaActual)) < 10**(-4) and abs(delta_T00/float(t00Actual)) < 10**(-4) and abs(delta_R00/float(r00Actual)) < 10**(-4) and abs(delta_T01/float(t01Actual)) < 10**(-3) :
            print("Bulk = ", BActual, "delta = ", delta_B)
            print("alpha = ", alphaActual, "delta = ", delta_Alpha)
            print("beta = ", betaActual, "delta = ", delta_Beta)
            print("t00 = ", t00Actual, "delta = ", delta_T00)
            print("r00 = ", r00Actual, "delta = ", delta_R00)
            encontrado = True 
    
    return BActual, alphaActual, betaActual, t00Actual, r00Actual, t01Actual

#############################################################################################################################################################

def optimizarParametrosMultiplicador_1fijo(r_exp, p_exp, betaActual, delta_Beta, t00Actual, delta_T00, r00Actual, delta_R00):
    error_anterior = errorMultiplicador(r_exp, p_exp)
    encontrado = False
    while not encontrado:

        CambiarBeta(betaActual, str(abs(float(betaActual) + delta_Beta)))
        
        error_beta = errorMultiplicador(r_exp, p_exp)
        error_beta = [error_beta, 'beta']

        CambiarBeta(str(abs(float(betaActual) + delta_Beta)), betaActual)

        CambiarT00(t00Actual, str(abs(float(t00Actual) + delta_T00)))

        error_t00 = errorMultiplicador(r_exp, p_exp)
        error_t00 = [error_t00, 't00']
        
        CambiarT00(str(abs(float(t00Actual) + delta_T00)), t00Actual)

        CambiarR00(r00Actual, str(abs(float(r00Actual) + delta_R00)))

        error_r00 = errorMultiplicador(r_exp, p_exp)
        error_r00 = [error_r00, 'r00']

        CambiarR00(str(abs(float(r00Actual) + delta_R00)), r00Actual)


        errores = [error_beta, error_t00, error_r00]
        menor_error = min(errores, key=lambda x: x[0])

        if menor_error[0] <= error_anterior:
         
            if menor_error[1] == 'beta':
                if float(betaActual)+delta_Beta <0:
                    CambiarBeta(betaActual, '0.001')
                    betaActual = '0.001'
                    delta_Beta = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarBeta(betaActual, str(float(betaActual)+delta_Beta))
                    betaActual = str(float(betaActual)+delta_Beta)
            
            elif menor_error[1] == 't00':
                if float(t00Actual)+delta_T00 <0:
                    CambiarT00(t00Actual, '0.001')
                    t00Actual = '0.001'
                    delta_T00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarT00(t00Actual, str(float(t00Actual)+delta_T00))
                    t00Actual = str(float(t00Actual)+delta_T00)
            
            elif menor_error[1] == 'r00':
                if float(r00Actual)+delta_R00 <0:
                    CambiarR00(r00Actual, '0.001')
                    r00Actual = '0.001'
                    delta_R00 = 10**(-12)
                else:
                    print("!!!!!!!!!!!!!!!!!!!")
                    CambiarR00(r00Actual, str(float(r00Actual)+delta_R00))
                    r00Actual = str(float(r00Actual)+delta_R00)


        if error_beta[0] >= error_anterior:
            delta_Beta = -0.97 * delta_Beta
        if error_t00[0] >= error_anterior:
            delta_T00 = -0.97 * delta_T00
        if error_r00[0] >= error_anterior:
            delta_R00 = -0.97 * delta_R00

        if menor_error[0]< error_anterior:
            error_anterior = menor_error[0]
        print("ERROR ANTERIOR = ", error_anterior)


        if abs(delta_Beta/float(betaActual)) < 10**(-5) and abs(delta_T00/float(t00Actual)) < 10**(-5) and abs(delta_R00/float(r00Actual)) < 10**(-5):
            print("beta = ", betaActual, "delta = ", delta_Beta)
            print("t00 = ", t00Actual, "delta = ", delta_T00)
            print("r00 = ", r00Actual, "delta = ", delta_R00)
            encontrado = True 
    
    return betaActual, t00Actual, r00Actual

#############################################################################################################################################################

def leer_archivoDATATXT(file_name):
    """Archivo EOS
    Columnas: "i","j","Rho","T","P","Pi","Pe","E","Ei","Ee" """

    # Initialize a list to store the data blocks
    data_blocks = []

    # Initialize a list to store the current data block
    current_block = []

    # Open the file for reading
    with open(file_name, 'r') as file:
        for line in file:
            if not line.startswith('#'):        # Strip leading and trailing whitespace from the line
                line = line.strip()

                if not line:
                    # If the line is blank, it's the end of the current data block
                    if current_block:
                        data_blocks.append(current_block)
                        current_block = []
                else:
                    # Split the line into columns (assuming columns are separated by whitespace)
                    columns = line.split()
                    current_block.append(columns)

    # If there is data left in the current_block at the end of the file, add it to data_blocks
    if current_block:
        data_blocks.append(current_block)

    # Print the data blocks

    """for i, block in enumerate(data_blocks, 1):
        print(f"Data Block {i}:")
        for row in block:
            print("\t".join(row))
        print("\n")"""

    a=np.array(data_blocks)
    indx=a[0,:,0]
    R=a[0,:,2].astype(float) 
    R *= 1000
    RR0 = R / R[0] 
    T=a[:,0,3].astype(float) #K
    P=a[:,:,4].astype(float) #GPa
    e=a[:,:,7].astype(float) #MJ/kg

    return R, RR0, T, P, e

#############################################################################################################################################################

def leer_archivoEOSTABLE(file_name):
    data_blocks = []
    current_block = []

    def convert_to_float(value):
        try:
            if value == 'NaN':
                return float('nan')
            elif value in ['Infinity', 'Inf']:
                return float('inf')
            elif value in ['-Infinity', '-Inf']:
                return float('-inf')
            else:
                return float(value)
        except ValueError:
            return value  # Devolver el valor original si no es un número

    with open(file_name, 'r') as file:
        for line in file:
            if not line.startswith(' #'):
                line = line.strip()

                if not line:
                    if current_block:
                        data_blocks.append(current_block)
                        current_block = []
                else:
                    columns = line.split()
                    columns = [convert_to_float(col) for col in columns]
                    current_block.append(columns)

    if current_block:
        data_blocks.append(current_block)

    # Convertir cada bloque a un DataFrame de pandas
    data_frames = [pd.DataFrame(block) for block in data_blocks if block]

    # Encontrar el número máximo de columnas en los bloques de datos
    max_columns = max(df.shape[1] for df in data_frames)
    
    # Normalizar el número de columnas agregando NaN donde falten columnas
    normalized_frames = []
    for df in data_frames:
        if df.shape[1] < max_columns:
            df = df.reindex(columns=range(max_columns), fill_value=np.nan)
        normalized_frames.append(df)
    
    # Concatenar todos los data frames en uno solo
    df = pd.concat(normalized_frames, ignore_index=True)
    
    # Convertir columnas específicas a float
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Extraer R y RR0 (vectores)
    R = data_frames[0].iloc[:, 2].astype(float).values
    R *= 1000
    RR0 = R / R[0]
    
    # Extraer T (vector)
    T = np.array([df.iloc[0, 3] for df in data_frames]).astype(float) * 11604
    
    # Determinar el número de bloques y el tamaño de cada bloque
    num_blocks = len(data_frames)
    num_rows_per_block = len(df) // num_blocks
    
    # Ajustar el tamaño esperado al número real de filas del DataFrame concatenado
    expected_size = num_blocks * num_rows_per_block
    if len(df) != expected_size:
        # Ajustar el número de bloques para coincidir con el tamaño del DataFrame
        num_rows_per_block = len(df) // num_blocks
        expected_size = num_blocks * num_rows_per_block
    
    # Extraer e, P, u, cs, cv, z (matrices)
    e = df.iloc[:expected_size, 4].astype(float).values.reshape(num_blocks, num_rows_per_block) * 10**(-10)
    P = df.iloc[:expected_size, 5].astype(float).values.reshape(num_blocks, num_rows_per_block) * 10**(-9)
    cs = df.iloc[:expected_size, 6].astype(float).values.reshape(num_blocks, num_rows_per_block) * 1e-2
    cv = df.iloc[:expected_size, 7].astype(float).values.reshape(num_blocks, num_rows_per_block) * 1e-7
    z = df.iloc[:expected_size, 8].astype(float).values.reshape(num_blocks, num_rows_per_block)  
    cond1 = df.iloc[:expected_size, -3].astype(int).values.reshape(num_blocks, num_rows_per_block)
    cond2 = df.iloc[:expected_size, -2].astype(int).values.reshape(num_blocks, num_rows_per_block)
    cond3 = df.iloc[:expected_size, -1].astype(int).values.reshape(num_blocks, num_rows_per_block)

    """cond1 = cond1.astype(str)
    cond2 = cond2.astype(str)
    cond3 = cond3.astype(str)


    combined_strings = np.core.defchararray.add(np.core.defchararray.add(cond1, cond2), cond3)

    shape = combined_strings.shape
    flat_list = combined_strings.flatten()
    int_list = [int(b) for b in flat_list]
    condiciones = np.array(int_list).reshape(shape)"""

    rows, cols = cond1.shape
    condiciones = np.empty((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            condiciones[i, j] = int(f"{cond1[i, j]}{cond2[i, j]}{cond3[i, j]}")
        
    return R, RR0, T, P, e, cs, cv, z, condiciones


#############################################################################################################################################################

def graf_P_RT_FEOS(path, cantidad, xlim, ylim):

    r, rr0, T, P, e = leer_archivoDATATXT(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7,7))

    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'$\mathrm{{P\ (GPa)}}$')
    ax.set_title(r'Tablas FEOS')

    idx_graficar = np.linspace(0, len(T) - 1, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)

    colores = plt.cm.winter(np.linspace(0, 1, len(T)))
    for i in range(len(T)-1):
        y = P[i,:]
        if i in idx_graficar:
            ax.plot(r,y, color = colores[i], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5)
    #ax.set_xlim(1,10**7)

    #f.config_logPlots(ax)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)

#############################################################################################################################################################

def graf_P_RT_EOSTABLE(path, cantidad, xlim, ylim):

    r, rr0, T, P, e, cs, cv, z, condiciones = leer_archivoEOSTABLE(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5,7))
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'$\mathrm{{P\ (GPa)}}$')
    #ax.set_title(r'Programa que llama la función eos')
    idx_graficar = np.linspace(40, len(T)-40, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)
    j = 0
    colores = plt.cm.winter(np.linspace(0, 1, cantidad))
    for i in idx_graficar:
        ax.plot(r,P[i,:], color = colores[j], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5)
        j += 1

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    #ax.set_xlim(min(r), max(r))

    #ax.set_title('Sin Multiplicador')
    ax.set_title('Poliestireno - Presión')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)
    return fig

#############################################################################################################################################################

def graf_e_RT_FEOS(path, cantidad, xlim, ylim):

    r, rr0, T, P, e = leer_archivoDATATXT(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7,7))

    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'$\mathrm{{e\ (MJ/kg)}}$')


    idx_graficar = np.linspace(0, len(T) - 1, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)

    colores = plt.cm.winter(np.linspace(0, 1, len(e)))
    for i in range(len(e)-1):
        y = e[i,:]
        if i in idx_graficar:
            ax.plot(r,y, color = colores[i], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1)
    #ax.set_xlim(1,10**7)

    #f.config_logPlots(ax)
    #ax.set_xlim(xlim[0], xlim[1])
    #ax.set_ylim(ylim[0],ylim[1])

    ax.set_title('Sin Multiplicador')
    #ax.set_title('Con Multiplicador')

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)

#############################################################################################################################################################

def graf_e_RT_EOSTABLE(path, cantidad, xlim, ylim):

    r, rr0, T, P, e, cs, cv, z, condiciones = leer_archivoEOSTABLE(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5,7))

    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'$\mathrm{{u\ (MJ/kg)}}$')


    idx_graficar = np.linspace(40, len(T) - 40, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)

    colores = plt.cm.winter(np.linspace(0, 1, cantidad))
    j = 0
    for i in idx_graficar:
        ax.plot(r,e[i,:], color = colores[j], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5)
        j += 1

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])

    ax.set_title('Poliestireno - Energía Interna')
    #ax.set_title('Con Multiplicador')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)
    return fig

#############################################################################################################################################################

def graf_z_RT_FEOS(path, cantidad, xlim, ylim):

    r,T, dataframe_ionizacion = procesar_archivo(path)
    kB = 8.617e-5
    T = T / kB

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7,7))

    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'Ionización')


    idx_graficar = np.linspace(0, len(T) - 1, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)

    colores = plt.cm.winter(np.linspace(0, 1, len(T)))
    for i in range(len(T)-1):
        y = dataframe_ionizacion.iloc[:,i].to_numpy(dtype = float) / 16
        if i in idx_graficar:
            ax.plot(r,y, color = colores[i], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1)
    #ax.set_xlim(1,10**7)

    #f.config_logPlots(ax)
    #ax.set_xlim(xlim[0], xlim[1])
    #ax.set_ylim(ylim[0],ylim[1])

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    #GRID
    ax.grid()

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    formatter.set_powerlimits((0, 1)) 
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=5.0,subs=(10.0, 5.0)))

    # Configurar las marcas (ticks) en el eje Y
    #ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    #ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    #ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)

#############################################################################################################################################################

def graf_cv_RT_EOSTABLE(path, cantidad, xlim, ylim):
    from matplotlib.patches import Patch

    r, rr0, T, P, e, cs, cv, z , condiciones= leer_archivoEOSTABLE(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5,7))
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'$\mathrm{{cv\ (J/kg K)}}$')
    #ax.set_title(r'Programa que llama la función eos')
    idx_graficar = np.linspace(40, len(T) - 40, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)

    colores = plt.cm.winter(np.linspace(0, 1, len(T)))
    for i in range(len(T)-1):
        y = cv[i,:]
        if i in idx_graficar:
            ax.plot(r,y, color = colores[i], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])

    #ax.set_title('Sin Multiplicador')
    ax.set_title('$\mathrm{{Poliestireno\ -\ c_v}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)

    #GRÁFICA CONDICIÓN Cv>0
    plt.figure(figsize=(10, 8))
    
    colores = ['black', 'white']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colores)

    cv_positivo = np.zeros_like(cv)  
    filas, columnas = cv.shape

    for i in range(filas):
        for j in range(columnas):
            if cv[i, j] > 0:
                cv_positivo[i, j] = 1
            else:
                cv_positivo[i, j] = 0

    X, Y = np.meshgrid(r+1e-6,T+1e-6)
    print(min(T))
    #GRAFICAR
    fig1, ax1 = plt.subplots(figsize = (7,7))
    ax1.pcolormesh(X,Y,cv_positivo, cmap = custom_cmap)

    #AJUSTES GRÁFICA
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-3,1e9)
    ax1.set_ylim(1e-4,1e9)
    ax1.set_ylabel(r'T (K)')
    ax1.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    legend_elements = [Patch(facecolor='white', edgecolor='black', label='$\mathrm{{c_v>0}}$'),
            Patch(facecolor='black', edgecolor='black', label='$\mathrm{{c_v<0}}$')]

    # Añadir la leyenda
    legend = ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=False, shadow=False, framealpha=1, borderpad=1, fontsize = 14)
    legend.get_frame().set_edgecolor('black')  
    legend.get_frame().set_linewidth(1) 
    return fig, fig1

#############################################################################################################################################################

def graf_cs_RT_EOSTABLE(path, cantidad, xlim, ylim):
    from matplotlib.patches import Patch

    r, rr0, T, P, e, cs, cv, z , condiciones= leer_archivoEOSTABLE(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5,7))
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$', fontsize = 12)
    ax.set_ylabel(r'$\mathrm{{cs\ (m/s)}}$', fontsize = 12)
    #ax.set_title(r'Programa que llama la función eos')
    idx_graficar = np.linspace(40, len(T) - 40, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)
    colores = plt.cm.winter(np.linspace(0, 1, cantidad))
    j = 0
    for i in idx_graficar:
        ax.plot(r,cs[i,:], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5, color = colores[j])
        j += 1
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])

    #ax.set_title('Sin Multiplicador')
    ax.set_title('$\mathrm{{Poliestireno\ -\ c_s}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize= 12)

    #GRID
    ax.grid()

    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)
    
    #GRÁFICA CONDICIÓN CS>0
    plt.figure(figsize=(10, 8))
    
    colores = ['black', 'white']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colores)

    cs_positivo = np.zeros_like(cs)  
    filas, columnas = cs.shape

    for i in range(filas):
        for j in range(columnas):
            if cs[i, j] > 0:
                cs_positivo[i, j] = 1
            else:
                cs_positivo[i, j] = 0

    X, Y = np.meshgrid(r+1e-6,T+1e-6)
    #GRAFICAR
    fig1, ax1 = plt.subplots(figsize = (7,7))
    ax1.pcolormesh(X,Y,cs_positivo, cmap = custom_cmap)

    #AJUSTES GRÁFICA
    #cax = ax1.imshow(cs_positivo, cmap='gray', vmin=0, vmax=1)
    #cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    #cbar.set_ticks([0, 1])    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-3,1e9)
    ax1.set_ylim(1e-4,1e9)
    ax1.set_ylabel(r'T (K)', fontsize = 12)
    ax1.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$', fontsize = 12)
    #ax1.legend(title='Negro = cs<0', loc = 'upper left')
    legend_elements = [Patch(facecolor='white', edgecolor='black', label='$\mathrm{{c_s>0}}$'),
                   Patch(facecolor='black', edgecolor='black', label='$\mathrm{{c_s<0}}$')]

    # Añadir la leyenda
    legend = ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=False, shadow=False, framealpha=1, borderpad=1, fontsize = 14)
    legend.get_frame().set_edgecolor('black')  
    legend.get_frame().set_linewidth(1) 

    return fig, fig1

#############################################################################################################################################################


def graf_z_RT_EOSTABLE(path, cantidad, xlim, ylim):
    from pprint import pprint

    r, rr0, T, P, e, cs, cv, z, condiciones = leer_archivoEOSTABLE(path)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8.5,7))
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')
    ax.set_ylabel(r'Ionización por átomo')
    #ax.set_title(r'Programa que llama la función eos')
    idx_graficar = np.linspace(40, len(T)-40, cantidad, endpoint=True)
    idx_graficar = np.round(idx_graficar).astype(int)
    colores = plt.cm.winter(np.linspace(0, 1, cantidad))
    j = 0
    for i in idx_graficar:
        ax.plot(r,z[i,:], color = colores[j], label=r'$\mathrm{{T\ =\ {:.0f}\ K}}$'.format(T[i]), linewidth=1.5)
        j += 1
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    #ax.set_xlim(min(r), max(r))

    #ax.set_title('Sin Multiplicador')
    ax.set_title('Poliestireno - Z')

    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #GRID
    ax.grid()

    """# Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    # Configuración simplificada del eje X
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='major')"""

    ax.grid(which='major', axis='both', linestyle='-', linewidth=1)
    return fig

#############################################################################################################################################################

def graf_Hugoniot(path):
    #Para graficar archivos de salida del programa hugoniot.F90

    datos_feos = leer_hugoniot(path)
    r_feos = datos_feos["R (kg/m3)"].to_numpy(dtype=float)
    p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
    T_feos = datos_feos["T (K)"].to_numpy(dtype=float) 
    Z_feos = datos_feos["z"].to_numpy(dtype=float)
    r_feos = r_feos[1:]
    p_feos = p_feos[1:]
    T_feos = T_feos[1:]#/11606
    Z_feos = Z_feos[1:]

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(r_feos,p_feos, label=r'Presión (GPa)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1),  ncol=1)
    ax.set_ylabel("Presión (GPa)", color = 'blue')
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')

    #ax.set_title('Sin Multiplicador')
    ax.set_title('Con Multiplicador')


    twin=ax.twinx()
    twin.plot(r_feos,T_feos, label=r'Temperatura (K)', color = 'orange')
    twin.legend(loc='upper left', bbox_to_anchor=(0, 0.95),  ncol=1)
    twin.tick_params(axis='y', labelcolor='orange')
    twin.set_ylabel("T (K)", color = 'orange')

    twin.set_xscale('log')
    twin.set_yscale('log')
    twin1 = ax.twinx()
    twin1.spines['right'].set_position(('outward', 60)) 
    twin1.plot(r_feos,Z_feos, label=r'Ionización por átomo', color = 'green')
    twin1.tick_params(axis='y', labelcolor='green')
    twin1.legend(loc='upper left', bbox_to_anchor=(0, 0.9),  ncol=1)
    twin1.set_ylabel("Ionización", color = 'green')

    twin1.set_xscale('log')


    """#Datos Multiplicador
    alpha = 0.044756778703152526
    beta = 1.544220733503368
    T00 = 115193.00579690476 * 11606
    R00 = 0.489475831525642 * 1000
    T01 = 1.1322851130325846 * 11606

    #Cálculo multiplicador
    alpha1 = 0.5 / alpha
    MEXP = (1-alpha1) * np.exp(-T_feos / T01)
    Tm = T00 * (r_feos / R00) * np.exp(1 - r_feos / R00)
    MULTIPLIER = 1 - alpha * (1 - MEXP) * np.exp(-(T_feos / Tm)**beta)

    twin2 = ax.twinx()
    twin2.spines['right'].set_position(('outward', 110)) 
    twin2.plot(r_feos,MULTIPLIER, label=r'Multiplicador', color = 'black')
    twin2.tick_params(axis='y', labelcolor='black')
    twin2.legend(loc='upper left', bbox_to_anchor=(0, 0.85),  ncol=1)
    twin2.set_ylabel("Multiplicador", color = 'black')

    twin2.set_xscale('log')"""

    """kB = 8.617333e-5
    n = r_feos/1000 * (Z_feos + 1) / (3.5 * 1.6726219e-24) #CH
    eF = 3.65e-15 * n ** (2/3)

    degeneracion = eF / (T_feos)

    twin2 = ax.twinx()
    twin2.spines['right'].set_position(('outward', 110)) 
    twin2.plot(r_feos,degeneracion, label=r'Degeneración', color = 'black')
    twin2.tick_params(axis='y', labelcolor='black')
    twin2.legend(loc='upper left', bbox_to_anchor=(0, 0.85),  ncol=1)
    twin2.set_ylabel("Degeneración", color = 'black')

    twin2.set_xscale('log')
    twin2.set_yscale('log')"""

    config_logPlots(ax)

    return ax

#############################################################################################################################################################

def graf_Hugoniot_showeos(path):
    #Para graficar archivos de salida del programa hugoniot.F90

    datos_feos = leer_archivo_hugoniot(path)
    r_feos = datos_feos["R (g/cm3)"].to_numpy(dtype=float) *1000
    p_feos = datos_feos["P (GPa)"].to_numpy(dtype=float)
    T_feos = datos_feos["T (K)"].to_numpy(dtype=float) 
    r_feos = r_feos[1:]
    p_feos = p_feos[1:]
    T_feos = T_feos[1:]

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(r_feos,p_feos, label=r'Presión (GPa)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1),  ncol=1)
    ax.set_ylabel("Presión (GPa)", color = 'blue')
    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$')

    twin=ax.twinx()
    twin.plot(r_feos,T_feos, label=r'Temperatura (K)', color = 'orange')
    twin.legend(loc='upper left', bbox_to_anchor=(0, 0.95),  ncol=1)
    twin.tick_params(axis='y', labelcolor='orange')
    twin.set_ylabel("T (K)", color = 'orange')

    twin.set_xscale('log')
    twin.set_yscale('log')


    config_logPlots(ax)

    return ax

#############################################################################################################################################################

def graf_condiciones(path):
    import matplotlib.colors as mcolors

    R, RR0, T, P, e, cs, cv, z, condiciones = leer_archivoEOSTABLE(path)
    plt.rc('font', family='serif')        

    fig, ax = plt.subplots(figsize=(10, 8))
    
    #colores = ['black', 'blue', 'cyan', 'magenta', 'red', 'yellow', 'orange', 'green']
    #custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colores)
    cmap = mcolors.ListedColormap(['black', 'darkred', 'purple', 'blue', 'orange', 'yellow', 'green'])
    bounds = [0, 1, 10, 11, 100, 110, 111, 112]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    X, Y = np.meshgrid(R, T)
    
    # Reemplazar valores no positivos en condiciones con el valor mínimo positivo
    #condiciones[condiciones <= 0] = np.min(condiciones[condiciones > 0])
    masked_data = np.ma.masked_where(~np.isin(condiciones, [0, 1, 10, 11, 100, 110, 111]), condiciones)

    cax = ax.pcolormesh(X, Y, condiciones, cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, ticks=[0, 1, 10, 11, 100, 110, 111])
    cbar.ax.set_yticklabels(['0', '1', '10', '11', '100', '110', '111'])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_title('Sin Multiplicador')
    ax.set_title('Poliestireno - Condiciones de Smith', fontsize = 14)
    ax.set_xlim(min(R), max(R))
    ax.set_ylim(min(T), max(T))
    
    # Personalizar las etiquetas para mostrar valores reales
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: '{:.1f}'.format(val)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: '{:.1f}'.format(val)))
    
    # Configurar las marcas (ticks) en el eje Y
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8), numticks=15))

    ax.set_xlabel(r'$\mathrm{{\rho\ (kg/m^3)}}$', fontsize = 14)
    ax.set_ylabel(r'T (K)', fontsize = 14)

    return fig