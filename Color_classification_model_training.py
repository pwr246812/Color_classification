import numpy as np


k = 3
h = 6
r = 5
gamma = 0.8
 
 
def czytaj(plik):
    data = np.genfromtxt(plik, delimiter=',')
    return data
 
 
def f_akt(x):
    return 1/(1 + np.exp(-x))
 
 
def skaluj(ciag_uczacy):
    ciag_uczacy[:,:3] = (ciag_uczacy[:,:3] - np.min(ciag_uczacy[:,:3]))/(np.max(ciag_uczacy[:,:3]) - np.min(ciag_uczacy[:,:3])) * (1 - (0)) + (0)
    return ciag_uczacy


def perceptron(tab_x, wagi):
    return f_akt(np.dot(tab_x, wagi[1:]) + wagi[0])
 
 
def hidden_layer(tab_x, wagi):
    hidden = [perceptron(tab_x, wagi[i, :k+1]) for i in range(h)]
    return hidden
 
 
def siec(tab_x, wagi):
    hidden = hidden_layer(tab_x, wagi)
    s = [perceptron(hidden, wagi[i, :h+1]) for i in range(h, h+r)]
    hidden.extend(s)
    return hidden
 
 
def policz_blad(ciag_uczacy, wagi):
    blad = 0
    for i in range(ciag_uczacy.shape[0]):
        s = siec(ciag_uczacy[i, :3], wagi)[h:r + h]
        if np.argmax(ciag_uczacy[i,3:]) != np.argmax(s):
            blad += 1
    return blad
 
 
def uczenie(ciag_uczacy, wagi):
    for i in range(ciag_uczacy.shape[0]):
        s = siec(ciag_uczacy[i, :k], wagi)
        # zewnętrzna
        for j in range(h, h + r):
            wagi[j, 0] -= gamma * 2 * (ciag_uczacy[i, j - h + k] - s[j]) * (-1) * s[j] * (1 - s[j]) * 1
            for z in range(1, h + 1):
                wagi[j, z] -= gamma * 2 * (ciag_uczacy[i, j - h + k] - s[j]) * (-1) * s[j] * (1 - s[j]) * s[z - 1]
        # wewnętrzna
        for j in range(h):
            suma = 0
            for p in range(h, h + r):
                suma += (ciag_uczacy[i, p - h + k] - s[p]) * (-1) * s[p] * (1 - s[p]) * wagi[p,j + 1]
            wagi[j, 0] -= gamma * 2 * suma * s[j] * (1 - s[j]) * 1
 
            for z in range(1, k+1):
                wagi[j,z] -= gamma * 2 *suma * s[j] * (1 - s[j]) * ciag_uczacy[i, z-1]
    return wagi

 
if __name__ == '__main__':
    x = czytaj('rgb.csv')
    ciag_uczacy = np.array([[i[0], i[1], i[2], 0, 0, 0, 0, 0] for i in x])
    for i in range(ciag_uczacy.shape[0]):
        ciag_uczacy[i, int(x[i][-1]) + 3] = 1
    ciag_uczacy = skaluj(ciag_uczacy)
    wagi = np.random.random_sample((11, 7))
    epoka = 0
    while policz_blad(ciag_uczacy, wagi):
        uczenie(ciag_uczacy, wagi)
        epoka += 1
    print(f"Uczenie zakończone pomyślnie! Liczba epok: {epoka}\n")

    print("double wagi[11][7] = {")
    for i in range(wagi.shape[0]):
        print('{ ', end='')
        for j in range(wagi.shape[1]):
            if j != wagi.shape[1]-1:
                print(wagi[i][j], end=', ')
            else:
                print(wagi[i][j], end='},')
        print()
    print("};")
    print("\nint ciag_obserwacji[25][4] = {")
    for i in range(x.shape[0]):
        print('{', end='')
        for j in range(x.shape[1]):
            if j != x.shape[1]-1:
                print(int(x[i][j]), end=', ')
            else:
                print(int(x[i][j]), end='},')
        print()
    print("};")