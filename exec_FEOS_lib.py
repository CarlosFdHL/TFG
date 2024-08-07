import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import subprocess
from scipy.optimize import curve_fit


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
                            palabras[i] = nueva_palabra
                            encontrado = True
                            break
                    nueva_linea = ' '.join(palabras)
                    file.write(nueva_linea + '\n')
                else:
                    file.write(linea)

        if not encontrado:
            print(f"No se encontró ninguna línea que comience con '{inicio_linea}' o la palabra '{palabra_a_editar}' no está presente.")
        else:
            print(f"La palabra '{palabra_a_editar}' en la línea que comienza con '{inicio_linea}' ha sido reemplazada por '{nueva_palabra}'.")
    except FileNotFoundError:
        print(f"El archivo '{path}' no fue encontrado.")

def ejec_showeos(elemento, operacion): #ejecutar el programa showeos para un elemento (ej:Al) y un modo de operación: 1- Isoterma, 2- Isócoras, 3-Isentrópicas, 4-Mountain-Plot, 5- Hugoniot, 6- Info. pto individual
    respuesta = subprocess.call(["/home/carlosfdhl/work/FEOS/run/showeos", elemento, operacion], cwd = "/home/carlosfdhl/work/FEOS/run" )
    print(respuesta)



def leer_archivo_hugoniot(path): #Lee el archivo en 'path' y devuelve el Data Frame 'datos_Al_hug' habiendo cambiado las unidades
    try:
        datos_Al_hug = pd.read_csv(path, sep="  ")

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

def func(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return b*((a-x)/(a*x-1))+c*x**d-e**(f*x)- g  + j*np.sin(k*x+l)+m*x**n+h*x**i

def reg_lineal(func, x, y, p0, a, err, max_iteraciones):
    sigma = y * err
    params, covariance = curve_fit(func, x, y, p0 = p0,
                            sigma = sigma,
                            absolute_sigma = True,  
                            maxfev = max_iteraciones)
    return params, covariance

def opt_a(x, y,func, err ): 
    opt_a=2 #valor mínimo de a
    max_r2=0 
    val_a=np.linspace(2,20,19)
    sigma=y*err 

    for i in val_a:
            try:
                    params, covariance = curve_fit(func, x, y,
                                    p0=[i,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                                    bounds=((i,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-0.1), 
                                            (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0.1)),
                                    sigma=sigma,
                                    absolute_sigma= True,  
                                    maxfev=20000)
                    residuals = y- func(x, *params)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    if r_squared>max_r2:
                            opt_a=i
                            max_r2=r_squared
            except:
                    pass

    print("El valor de a que más se ajusta a los datos es: a= ",opt_a)
    return opt_a

def graf_fit(x, y, nptos, params, nombre_eje_x, nombre_eje_y, titulo, escalax, escalay ):

    residuals = y- func(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print ('r2 = ', r_squared)

    x_fit = np.linspace(min(x), max(x), nptos)
    y_fit = func(x_fit,*params) 

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()  

    ax.set_xlabel(nombre_eje_x)
    ax.set_ylabel(nombre_eje_y)
    ax.set_title(titulo)

    p1=ax.scatter(x,y, label = "P (GPa)")
    ax.plot(x_fit, y_fit, color='red', label='Ajuste: P / $\mathrm{r^2} = %.3f $' %(r_squared), ls="--", linewidth=2)

    ax.grid()
    ax.set_xscale(escalax)
    ax.set_yscale(escalay)
    ax.legend()
    plt.show()

#EJEMPLO USO: editar_palabra_en_archivo()
"""
archivo = 'FEOS_Material-DB.dat'
inicio_linea = '[3717]_Bulk'
palabra_a_editar = '7.5e11'
nueva_palabra = '7.5e11'
#nueva_palabra = float(palabra_a_editar)*1.05
#nueva_palabra = "{:e}".format(nueva_palabra)
#nueva_palabra = nueva_palabra.replace("+", "")
path = ("/home/carlosfdhl/work/FEOS/EOS-Data/FEOS_Material-DB.dat")
editar_palabra_en_archivo(path, inicio_linea, palabra_a_editar, nueva_palabra)
"""


#EJEMPLO DE USO: ejec_showeos()
"""
elemento = 'Al'
operacion = '5'
ejec_showeos(elemento, operacion)
"""

#EJEMPLO DE USO: leer_archivo_hugoniot()
path = "/home/carlosfdhl/work/FEOS/run/Al.hug"
datos_feos = leer_archivo_hugoniot(path)

#EJEMPLO DE USO: opt_a() 
err = 0.044
max_iteraciones = 100000
a = 2
x = datos_feos['R/R0'].to_numpy(dtype = float)
y = datos_feos['P (GPa)'].to_numpy(dtype = float)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()  
ax.scatter(x,y)
plt.show()
#a = opt_a(x, y,func, err )

#EJEMPLO DE USO: reg_lineal()
p0 = [a, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
bounds=((a, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf ,-np.inf ,-np.inf ,-np.inf ,-np.inf ,-np.inf ,-np.inf ,-0.1), 
        (np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,1 ,np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,np.inf ,0.1))
#params , covariance = reg_lineal(func, x, y, p0, a, err, max_iteraciones)

#EJEMPLO DE USO: graf_fit()
nptos = 100
nombre_eje_x = 'R/R0'
nombre_eje_y = 'P (GPa)'
titulo = 'Aluminio'
escalax = 'log'
escalay = 'log'
#graf_fit(x, y, nptos, params, nombre_eje_x, nombre_eje_y, titulo, escalax, escalay )
