import numpy as np
JUDGES = 5


filename = "ex8_data.txt"
n = None                    #competitors
array = None                #array of score, for competitor i, array[i] contains their 5 scores
compNames = []              #names of competitors
with open(filename, "r") as f:
    lines = f.readlines()
    n = int(lines[0].strip())
    array = np.zeros(n, dtype=np.float32)   #array of competitors' sum of scores
    
    for i in range(1, len(lines)):
        #put every row of scores (ignore names country etc)
        tempArray = np.array([float(i) for i in lines[i].strip().split(" ")[3:]])  #array[i-1, :]
        tempArray = np.delete(tempArray, np.argmax(tempArray))
        tempArray = np.delete(tempArray, np.argmin(tempArray))
        array[i-1] = np.sum(tempArray)
        name = lines[i].split(" ")[0] + lines[i].split(" ")[1]
        compNames.append(name)

    #array = array.reshape(array.shape[1], array.shape[0]) -> prima che era colonna bisognava fare questo, ora fin dall'inizio è un row array quindi non devi farlo perchèè già così
    sortedArgs = (-array).argsort()   #negate the array since it's the only way to have a descending sort
    print(array)
    print(sortedArgs)
    print()
    counter = 0
    for i in range(3):     #(1, n)
        print(f"{i}. {compNames[sortedArgs[i]]} - Score: {array[sortedArgs[i]]}")
        


    


