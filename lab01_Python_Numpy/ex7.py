import numpy as np

#a

def function_a(m, n):
    #The most efficient solution uses numpy broadcasting without cycles!

    #m -> rows
    #n -> cols

    rows = np.arange(m) #shape is (m, ) -> it's a ROW ARRAY -> (1, m)
    cols = np.arange(n) #shape is (n, ) -> it's a ROW ARRAY -> (1, n)

    #in order to multiply them I have to make one of them as a column array of shape (x, 1) where x can be either m or n -> column array with more rows
    rows = rows.reshape((m, 1))

    res = rows * cols

    return res


m = 3
n = 4
print(f"a) {function_a(m, n)}")



#b

x = np.array([[1, 2, 6, 4],[3, 4, 3, 7], [1, 4, 6, 9]], dtype=np.float32)


def normalize_cols(x):
    #input: 2d numpy array x
    #normalize each column element
    cols_sum = x.sum(axis=0)

    #in the example we have:
    #np.shape(x) -> (3, 4)
    #np.shape(cols_sum) -> (4, )
    #I can do x / y to divide x by y element-wise

    return x / cols_sum

print(f"b1) {normalize_cols(x)}")


#same input x as before

def normalize_rows(x):
    #normalize each row element based on the sum of all the elements of that row
    rows_sum = x.sum(axis = 1)

    #np.shape(x) -> (3, 4)
    #np.shape(rows_sum) -> (3, )

    return x / rows_sum.reshape((x.shape[0], 1))

print(f"b2) {normalize_rows(x)}")


#d

x1 = np.array([[1, -12, 6, -4],[3, 4, 3, -7], [1, 4, 6, 9]], dtype=np.float32)

def setNegativesToZero(x):
    return np.maximum(x, 0) #element-wise maximum between each element of x and 0

print(f"c) {setNegativesToZero(x1)}")

#e

matrix1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
matrix2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

def sumofProduct(m1, m2):
    #scalar product
    p = np.dot(m1, m2)  #m1 @ m2
    return p.sum()


print(f"d) {sumofProduct(matrix1, matrix2)}")




    



