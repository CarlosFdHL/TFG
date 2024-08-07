
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
import pandas as pd
# Autor (c): Pedro Velarde, Universidad Politécnica de Madrid, marzo 2024
# Preparado para TFG de Carlos Fdez de Heredia
# version 1.0   22 abril 2024

def p2(a1,a2,b0,b1,b2,x):
    p2=(a1*x+a2*x*x)/(b0+b1*x+b2*x*x)
    return p2
def p3(a2,a3,b0,b1,b2,b3,x):
    p3=(a2+a3*x)/(b0+b1*x+b2*x*x+b3*x*x*x)*x*x
    return p3
def p3a(a1,a3,b0,b1,b2,b3,x):
    p3=(a1+a3*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x)*x
    return p3
def pk1(a,b,n,x):
    p=a*x**n/(np.exp(b*x)-1)
    return p
def pk2(a,b,n,m,x):
    p=a*x**n*np.exp(-b*x**m)
    return p
def pk3(a,b,n,m,x):
    p=1-1/(np.exp(a*(x-b))+1)
    return p
def pk4(a,b,n,m,x):
    p=1-np.exp(-b*x**m)
    return p
def fa(n,m,s,xm,ym,g,x):
    b=n/m/xm**m
    a=ym/xm**n*np.exp(n/m)
    p=pk2(a,b,n,m,x)
    q=pk4(a,b,n,m,x)  
    p=g*q**s+p
    return p,q


# Lee datos de filename saltando comentarios
def read_data(filename):
    try:
        names = ["R (g/cm3)", "T (K)",     "P (GPa)", 
                 "E (kJ/kg)", "Us (km/s)", "Up (km/s)"]
        fac_units=np.array(
                [1.0,         11606.0,      100.0,
                 1e-7,         1.0,         1.0 ]) # conversión de unidades
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
    pres=data[[namex]].to_numpy(dtype=float)
    rho=data[[namey]].to_numpy(dtype=float)
    pres=pres/pres[0]
    rho=rho/rho[0]
    linearp=False
    if linearp:
        pres=pres-1
    else:
        pres=np.log(pres)

    linearrho=True
    if linearrho:
        rho=rho-1
        g=3
    else:
        rho=np.log(rho)
        g=np.log(4)
    k=np.argmax(rho)
    xm=pres[k]
    ym=rho[k]
    print("MAXIMOS:",xm,ym)
    x=pres
    poly=False
    if poly:
        b2a=np.linspace(1,20000,5)
        b1a=np.linspace(0,0,5)
        if linearp:
            plt.xscale("log")
            for b2 in b2a:
                for b1 in b1a:
                    a2=g*b2
                    a1=ym*b1+2*(ym-g)*xm*b2
                    b0=(ym-g)/ym*xm*xm*b2
                    p=p2(a1,a2,b0,b1,b2,x)
                    plt.plot(pres,p)

        plt.plot(pres,rho,'.r')
        plt.show()

        b0=xm**3*(ym-g)/(2*ym)
        a2=3*ym*b0/xm**2
        a3=g
        b2=0
        b1=0
        b3=1
        p=p3(a2,a3,b0,b1,b2,b3,x)
        if linearp:
            plt.xscale("log")

        plt.plot(pres,p)
        plt.plot(pres,rho,'.r')
        plt.show()


        b0=2*xm**3*(ym-g)/(ym)
        a1=3*ym*b0/(2*xm)
        a3=g
        b2=0
        b1=0
        b3=1
        p=p3a(a1,a3,b0,b1,b2,b3,x)
        if linearp:
            plt.xscale("log")
        plt.plot(pres,p)
        plt.plot(pres,rho,'.r')
        plt.show()
        from scipy.optimize import root_scalar
        def f(x):
            return((x/n-1)*np.exp(x)+1)
        for y in range(10):
            y=0.3*y
            print(y,f(y))
        t=root_scalar(f, bracket=[0.1, 4])
      
        a=ym*(np.exp(t.root)-1)/xm**n
        b=t.root/xm
        print("X=",t.root,a,b)
        p=pk1(a,b,n,x)
    else:
        # Chebychev
        print("x=?",x.shape)
        xp=pres[:,0]
        yr=rho[:,0] # cheby no admite 2d, solo 1d, y pres es 2d
        coeftchb=np.polynomial.chebyshev.chebfit(xp,yr,8)
        polytchb=np.polynomial.chebyshev.chebval(xp,coeftchb)
        errort=sum((yr-polytchb)**2)/rho.size
        # Fit Propio
        error=np.sum(rho**2)/rho.size
        nopt=1
        mopt=1
        xmopt=xm
        sopt=3
        N=20       # Número de puntos a probar, crece como N^4
        for n in np.linspace(1.28,1.29,N):
            for m in np.linspace(3.7,3.71,N):
                for xmo in np.linspace(0.87*xm,0.871*xm,N):
                    for s in np.linspace(3.43,3.45,N):
            #n=1.3
            #m=3
                        p,q=fa(n,m,s,xmo,ym,g,x)
                        errorl=np.sum((rho-p)**2)/rho.size
                        if errorl<error:
                            nopt=n
                            mopt=m
                            xmopt=xmo
                            sopt=s
                            error=errorl
        p,q=fa(nopt,mopt,sopt,xmopt,ym,g,x)
        if linearp:
            plt.xscale("log")

        print("NOPT=",nopt,mopt,sopt,xmopt/xm,"e=",error," etchb=",errort)

        plt.plot(pres,rho,'.r')
        #plt.plot(pres,q,"red")
        plt.plot(pres,p,"blue",linewidth=2)
        plt.plot(pres,polytchb,"green")

        plt.show()
# parte principal cuando no como módulo
if __name__ == "__main__":
    filename="/home/carlosfdhl/work/FEOS/run/Al.hug"
    data=read_data(filename)
    # Curva P vs rho que sale BIEN
    setmodel("P (GPa)","R (g/cm3)",data)
    # Curva T vs rho que sale MAL
    #setmodel("R (g/cm3)","T (K)",data)
    # Curva P vs T que sale BIEN
    # Claramente el procedimiento va de rho->P->T
    #setmodel("P (GPa)","T (K)",data)
    

