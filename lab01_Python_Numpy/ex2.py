import sys
from math import sqrt
filename = sys.argv[1]
flag = sys.argv[2]
param = sys.argv[3]


def calcDist(points):
    #points: list of (X, Y) points
    #calculate the total euclidean distance traveled
    distance = 0
    for p in range(len(points) - 1):
        x1, y1 = points[p]
        x2, y2 = points[p+1]
        distance += sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

    return distance


if flag == "-b":
    #print total distance traveled by the bus determined by the busid additional param
    points = []
    more1 = 0
    with open(filename, "r") as f:
        for line in f:
            if line.split(" ")[0] == param:
                x = int(line.split(" ")[2])
                y = int(line.split(" ")[3])
                points.append((x, y))
                if (more1 == 0): more1 = 1

    if more1 > 0:
        distance = calcDist(points)
        print(f"{param} - Total Distance: {distance}")
    else: print(f"{param} - No Distance traveled!")

elif flag == "-l":
    #print avg speed of buses traveling on the line determined by the lineid additional param
    buses = {}  #key: busId, value = ([points], [time])
    totalTime = 0
    with open(filename, "r") as f:
        for line in f:
            if line.split(" ")[1] == param:
                x = int(line.split(" ")[2])
                y = int(line.split(" ")[3])
                busId = line.split(" ")[0]
                if busId not in buses:
                    buses[busId] = [[], []]

                buses[busId][0].append((x, y))
                buses[busId][1].append(int(line.split(" ")[4]))
                

    speeds = []
    n = 0
    for bus in buses:
        #calculate avg speed for each bus selected
        distance = calcDist(buses[bus][0])
        deltaTime = abs(buses[bus][1][-1] - buses[bus][1][0])
        speeds.append(float(distance) / float(deltaTime))
        n+=1

    s = 0
    for i in speeds: s+=i
    avgSpeed = float(s) / float(n)
    print(f"{param} - Avg Speed: {avgSpeed}")

    

      


        