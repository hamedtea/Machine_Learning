# Linear solver
#hamed.talebian@tun.fi - 150360360

from ast import main
import matplotlib.pyplot as plt
import numpy as np

def my_linefit(x,y): 
    
    """a function that calculate the gradient coefficient of one space linear regression problem
        param: x:list, y:list
        return: a:int, b:int 
    """
    x_bar = x.sum() / len(x) #normalized mean
    y_bar = y.sum() / len(y) #normalized mean
    xy = x*y 
    xy_bar = xy.sum() / len(x) * len(y) #normalized cross variance
    x_two = x*x 
    x_square_bar = x_two.sum() / len(x)*len(x) #normalized covariance

    a = (x_bar*y_bar - xy_bar) / ((x_bar*x_bar) - x_square_bar) #slope
    b = y_bar - (x_bar)*((x_bar*y_bar - xy_bar) / ((x_bar)*(x_bar) - x_square_bar)) #intercept
    return a , b



def main(): 
    x = np.random.uniform(-2 ,5 ,10) #random array between -2, 5, 10 points
    y = np.random.uniform(0 ,3 ,10)  
    a , b = my_linefit(x , y)
    plt.plot(x , y , 'kx' )
    xp = np.arange(-2 ,5 ,0.1)
    plt.plot(xp , a*xp+b , 'r' )
    print (f'My fit : a={a} and b={b}')
    plt.show()

if __name__ == "__main__":
    main()