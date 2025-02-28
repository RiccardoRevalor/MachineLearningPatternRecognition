import os

filename = "ex1_data.txt"

def computeScore(l):
    # remove the max and the min from the string l and compute score
    l = l.split(" ")[3::]  # remove the first 3 elements since they are not numerical values
    l = [float(i) for i in l]  # convert the list of strings to a list of integers
    l.remove(max(l))
    l.remove(min(l))
    # add the score to the line 
    return sum(l)

competitors = []
with open(filename) as f:
    for line in f:
        score = computeScore(line)
        line += f" {score}"
        competitors.append(line)


#sort the competitors by score
competitors.sort(key=lambda l: float(l.split(" ")[-1]), reverse=True)

#print final ranking of the 3 best competitors
top3 = competitors[:3]
for i in range(3):
    fields = top3[i].split(" ")
    print(f"1: {fields[0]} {fields[1]} - Score: {fields[-1]}")

#find country with the best score
countryScore = {}

for line in competitors:
    country = line.split(" ")[2]
    score = float(line.split(" ")[-1])
    if country not in countryScore:
        countryScore[country] = score
    else:
        countryScore[country] += score

topCountry = ""
maxScore = -1

for c in countryScore:
    if countryScore[c] > maxScore:
        maxScore = countryScore[c]
        topCountry = c

print(f"Best Country:\n{topCountry} - Best Score: {maxScore}")
