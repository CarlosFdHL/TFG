
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
import pysindy as ps
import pandas as pd
# Autor (c): Pedro Velarde, Universidad Politécnica de Madrid, marzo 2024
# Preparado para TFG de Carlos Fdez de Heredia
# version 1.0   22 abril 2024


def alpha(z,d):
    zd=z**(2*d)
    z2=z*z
    return (z2/(z2-d)/zd)
# obtener ceros z de d/(z^2-d)*exp(-z*z)-rhs
def zset(z,d,rhs): #rhs=(y0-g)/g>0
    zset=d/(z*z-d)*np.exp(-z*z)-rhs
    #print("zset=",zset,z,d)
    return zset
def fh(a,b,c,d,m,x):
    y=x**m
    z=np.exp(-b*y)
    fh=a*(1-z)+c*z*y**d
    return fh



# Lee datos de filename saltando comentarios
def read_data(filename):
    try:
        """
        #Nombres para el formato Al.hug de FEOS directamente
        names = ["R (g/cm3)", "T (K)",     "P (GPa)", 
                 "E (kJ/kg)", "Us (km/s)", "Up (km/s)"]
        fac_units=np.array(
                [1.0,         11606.0,      100.0,
                 1e-7,         1.0,         1.0 ]) # conversión de unidades
        """

        names = ["i","R (g/cm3)", "T (K)", "e (kJ/kg)" , "P (GPa)",
                "u(i)", "u(i)/cs", "dedr", "dpdr", "j-1", "error"]
        fac_units=np.array(
               [1.0, 1.0,  11606.0,  1e-7,    1e-9,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ]) # conversión de unidades
        datos=pd.read_csv(filename,
                          sep='\s+',
                          names=names,
                          comment='#',
                          )
        datos=datos.apply(pd.to_numeric, errors='coerce')*fac_units
        return datos
    except Exception as e:
        print(f"Error en fichero {filename}:", e)
#
def setmodel(namex,namey,data):
    pres=data[[namex]].to_numpy(dtype=float)[1:]
    rho=data[[namey]].to_numpy(dtype=float)[1:]
    rho0=data[[namey]].to_numpy(dtype=float)[0]
 
    gamma=5.0/3.0
    g=(gamma+1)/(gamma-1)*rho0/rho[0]
    pres=pres/pres[0]
    rho=rho/rho[0]
    print("g=",g)
    linearp=False
    linearrho=True

    poly=False

    if linearp:
        pres=pres-1
        plt.xscale("log")
    else:
        pres=np.log(pres)

    if linearrho:
        rho=rho-1
        g=g-1
    else:
        rho=np.log(rho)
        g=np.log(g)
    k=np.argmax(rho)
    xm=pres[k]
    ym=rho[k]
    print("MAXIMOS:",xm,ym)
    x=pres
    if poly:
        # Chebychev
        xp=pres[:,0]
        yr=rho[:,0] # cheby no admite 2d, solo 1d, y pres es 2d
        coeftchb=np.polynomial.chebyshev.chebfit(xp,yr,8)
        polytchb=np.polynomial.chebyshev.chebval(xp,coeftchb)
        errorchb=sum((yr-polytchb)**2)/rho.size
    else:
        # Fit Propio
        rhs=(ym-g)/g
        a=g
        error=np.sum(rho**2)/rho.size
        N=100       # Número de puntos a probar, crece como N^4
        for m in np.linspace(1,5,N):
            for d in np.linspace(0.1,1,N):
                ll=1.00001*np.sqrt(d)
                lr=np.sqrt(d*(1+1/rhs))
                sol = optimize.root_scalar(zset, args=(d,rhs),bracket=[ll,lr], method='brentq')
                z0=sol.root
                al=alpha(z0,d)
                b=z0*z0/xm**m
                c=a*al*b**d
                p=fh(a,b,c,d,m,x)
                errorl=np.sum((rho-p)**2)/rho.size
                if errorl<=error:
                    mopt=m
                    dopt=d
                    copt=c
                    bopt=b
                    error=errorl
        p=fh(a,bopt,copt,dopt,mopt,x)

        plt.plot(pres,rho,'.r')
        plt.plot(pres,p,"blue",linewidth=2)
        #plt.plot(pres,polytchb,"green")
        print("error,a,b,copt,dopt,mopt",error,a,b,copt,dopt,mopt)
        plt.show()
        print(p[-1])
# parte principal cuando no como módulo
if __name__ == "__main__":
    filename="/home/carlosfdhl/work/hugoniot/prueba1"
    data=read_data(filename)
    # Curva P vs rho que sale BIEN
    setmodel("P (GPa)","R (g/cm3)",data)
    # Curva T vs rho que sale MAL
    #setmodel("R (g/cm3)","T (K)",data)
    # Curva P vs T que sale BIEN
    # Claramente el procedimiento va de rho->P->T
    #setmodel("P (GPa)","T (K)",data)
    

