import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def lsfit(data):
    x = data[:,0]
    y = data[:,1]
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    xy_bar = np.mean(x*y)
    xx_bar = np.mean(x*x)

    w1 = (xy_bar - y_bar*x_bar)/(xx_bar - x_bar*x_bar)
    w0 = y_bar - w1*x_bar
    loss = np.mean((y - w0 - w1*x)*(y - w0 - w1*x))
    
    return w0, w1, loss

y = lambda x, w0, w1: w0 + w1*x

def lab1_scalar(data_m100, data_f100):
    plt.figure(0)
    plt.scatter(data_m100[:,0], data_m100[:,1], color='blue')
    x = data_m100[:,0]
    w0, w1, loss = lsfit(data_m100)
    plt.plot(x,y(x,w0,w1), color='blue')
    
    plt.hold(True)

    plt.scatter(data_f100[:,0], data_f100[:,1], color='red')
    x = data_f100[:,0]
    w0, w1, loss = lsfit(data_f100)
    plt.plot(x,y(x,w0,w1), color='red')


    plt.legend(('Mens', 'Womens'))
    plt.grid()
    plt.title('Times on 100 meter sprint')
    plt.xlabel('Year')
    plt.ylabel('Time [s]')
    plt.savefig('sprint100m.eps')


    plt.figure(1)
    x = np.linspace(2575, 2625, 20000)
    
    w0, w1, loss = lsfit(data_m100)
    y_m = y(x,w0,w1)
    plt.plot(x, y_m, color='blue')
    
    plt.hold(True)
    
    w0, w1, loss = lsfit(data_f100)
    y_f = y(x,w0,w1)

    plt.plot(x, y_f, color='red')

    idx = np.argwhere(np.diff(np.sign(y_m - y_f)) != 0).reshape(-1)
    plt.scatter(x[idx], y(x,w0,w1)[idx], marker='x')
    plt.text(x[idx]+3, y(x,w0,w1)[idx], 'Year: ' + str(round(float(x[idx]),2)))
    
    plt.legend(('Mens', 'Womens'))
    plt.grid()
    plt.title('Intersection of times on 100 meter sprint')
    plt.xlabel('Year')
    plt.ylabel('Time [s]')
    plt.savefig('sprint100m_intersection.eps')

size    = lambda mat: (np.size(mat, 0), np.size(mat, 1))
inv     = lambda mat: np.linalg.inv(mat)

def least_squares(data, poly_degree):
    x = np.matrix([data[:,0]]).transpose()
    t = np.matrix([data[:,1]]).transpose()

    X = np.power(x, 0)

    for i in range(1, poly_degree + 1):
        X = np.concatenate((X, np.power(x, i)), axis=1)

    X_T = X.transpose()

    return np.asarray((inv(X_T * X) * X_T * t).transpose())[0]

def fitline(data, poly_degree):
    coefficients = least_squares(data, poly_degree)
    x = data[:,0]
    y = [0] * len(x) #Preallocation
    for i in range(0, len(coefficients)):
        for j in range(0, len(x)):
            y[j] += coefficients[i] * np.power(x[j], i)
    return x, y

def compute_loocv(data, poly_degree): #Leave-one-out cross validation  
    err = 0

    for i in range(0, data.shape[0]):
        data_row_removed = np.delete(data, i, 0) #remove i-th element
        y_test = data_row_removed[:,1]
        _, y_fitline = fitline(data_row_removed, poly_degree)

        err += (y_test - y_fitline) * (y_test - y_fitline)
    
    err = np.mean(err)

    return err

def lab1_vector(data_m100, data_f100):
    [x, y] = fitline(data_f100, 2)
    
    plt.figure(0)
    plt.scatter(data_f100[:,0], data_f100[:,1])
    plt.plot(x,y)
    plt.grid()

    poly_err = []
    
    NUM_POLINOMIALS = 10

    for i in range(1, NUM_POLINOMIALS + 1):
        poly_err.append(compute_loocv(data_f100, i))

    plt.figure(1)
    plt.plot(np.arange(NUM_POLINOMIALS), poly_err)
    
    print(np.arange(NUM_POLINOMIALS))
    print(poly_err)
    
    plt.show()


def main():

    data = sio.loadmat('olympics.mat')
    data_m100 = data['male100']
    data_f100 = data['female100']
    data_m200 = data['male200']
    data_f200 = data['female200']
    data_m400 = data['male400']
    data_f400 = data['female400']

    #lab1_scalar(data_m100, data_f100)
    lab1_vector(data_m400, data_f400)

    #a = np.matrix('1 2; 3 4') * np.matrix('1; 2')

    #print(a)

main()