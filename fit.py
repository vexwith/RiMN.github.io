import numpy as np
from scipy.optimize import least_squares
import pylab as py

#Dane
daty =np.array([-40000,-2650,1500,1870,2050])
godziny=np.array([3,9.5667,15,19.5,20.5])
intervals_x=np.array([(-100000,-20000),(-2950,900),(1450,1600),(1810,1905),(2040,2100)])

X=np.linspace(-120000, 30000,50000)



#definicja funkcji
def arctan(x,a):
    return a[0]+a[1]*np.arctan(a[2]*x+a[3])

def der_arctan(x,a):
    return a[1]*a[2]/(1+(a[2]*x + a[3])**2)

def sin(x,a):
    return a[0]+a[1]*np.sin(a[2]*x+a[3])

def der_sin(x,a):
    return a[1]*a[2]/(1+(a[2]*x + a[3])**2)


def residuals(a):
    # a=vars[:4]
    # xs=vars[4:4+len(godziny)]
    # xs=vars[4:]
    return sin(daty,a) - godziny
    # fvals = arctan(xs,a)
    fvals = sin(xs,a)
    res = fvals - godziny
    # kara za ujemną pochodną (dla monotoniczności) <-- to robił chat, nie do końca wiem o co chodzi jeszcze
    grid = np.linspace(np.min(intervals_x[:,0]), np.max(intervals_x[:,1]), 100)
    dvals = der_sin(grid,a)
    penalty = np.sqrt(100.0) * np.minimum(0, dvals)
    return np.concatenate([res, penalty])

#initial guess
# p0 = np.array([10.0, 10.0, 0.001, 0.0])
p0 = np.array([12,     # poziom
               10,     # amplituda
               2*np.pi/10000,   # okres około 10 tys lat
               0.0])

vars0 = np.concatenate([p0, daty])




#ograniczenia <-- też podsunięte przez chat, no ale tu sens jest klarowniejszy
lower = np.concatenate([[0.0, 0.0, -np.inf, -np.inf],intervals_x[:,0]])
upper = np.concatenate([[np.inf, np.inf, np.inf, np.inf],intervals_x[:,1]])

# res=least_squares(residuals,vars0,bounds=(lower,upper),max_nfev=20000)
res=least_squares(residuals,p0)
a_opt=res.x[:4]
# x_opt=res.x[4:4+len(godziny)] #<-- to też dopisywał chat, nie jestem pewien, czym się to różni od dat tak po prostu

print("Params:", a_opt)


y_vals = sin(X, a_opt)

py.figure()
py.plot(X, y_vals, label="dopasowana funkcja arctan")
py.scatter(daty, godziny, color='red', label='punkty (x*, y)')
py.grid()
py.show()