filename = "ex5_data.txt"

room = None
with open(filename) as f:
    firstLine = 0
    for line in f:
        if firstLine == 0:
            firstLine = 1
            n = int(line)
            matrix = [[0.0] * n for i in range(n)]    #matrix nxn

        else:

            #read light sources
            x = int(line.split(" ")[0])
            y = int(line.split(" ")[1])

            matrix[x][y] += 1.0

            #8 adjacent tiles
            if x > 0:
                matrix[x-1][y]+=0.5
            if x < n-1:
                matrix[x+1][y]+=0.5
            if y > 0:
                matrix[x][y-1]+=0.5
            if y < n -1:
                matrix[x][y+1]+=0.5
            if x > 0 and y > 0:
                matrix[x-1][y-1] += 0.5  # diagonale in alto a sinistra
            if x > 0 and y < n - 1:
                matrix[x-1][y+1] += 0.5  # diagonale in alto a destra
            if x < n - 1 and y > 0:
                matrix[x+1][y-1] += 0.5  # diagonale in basso a sinistra
            if x < n - 1 and y < n - 1:
                matrix[x+1][y+1] += 0.5  # diagonale in basso a destra

            # 16 vicini (inclusi diagonali a distanza 2)
            if x > 1:
                matrix[x-2][y] += 0.2  # sopra 2 passi
            if x < n - 2:
                matrix[x+2][y] += 0.2  # sotto 2 passi
            if y > 1:
                matrix[x][y-2] += 0.2  # sinistra 2 passi
            if y < n - 2:
                matrix[x][y+2] += 0.2  # destra 2 passi
            if x > 1 and y > 1:
                matrix[x-2][y-2] += 0.2  # diagonale in alto a sinistra 2 passi
            if x > 1 and y < n - 2:
                matrix[x-2][y+2] += 0.2  # diagonale in alto a destra 2 passi
            if x < n - 2 and y > 1:
                matrix[x+2][y-2] += 0.2  # diagonale in basso a sinistra 2 passi
            if x < n - 2 and y < n - 2:
                matrix[x+2][y+2] += 0.2  # diagonale in basso a destra 2 passi

            if x > 0 and y > 1:
                matrix[x-1][y-2] += 0.2  # diagonale in alto a sinistra 1 passo sopra
            if x > 0 and y < n - 2:
                matrix[x-1][y+2] += 0.2  # diagonale in alto a destra 1 passo sopra
            if x < n - 1 and y > 1:
                matrix[x+1][y-2] += 0.2  # diagonale in basso a sinistra 1 passo sotto
            if x < n - 1 and y < n - 2:
                matrix[x+1][y+2] += 0.2  # diagonale in basso a destra 1 passo sotto



if matrix is not None: 
    print(matrix)
else: print("Empty room")


