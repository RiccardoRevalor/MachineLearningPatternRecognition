filename = "ex3_data.txt"

citiesBirths = {}
monthsBirths = {}
totBirths = 0

monthNames = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December"
}


with open(filename, "r") as f:
    for line in f:
        #<name> <surname> <birthplace> <birthdate>
        fields = line.split(" ")
        city = fields[2]
        month = monthNames[fields[3].split("/")[1]]
        if city not in citiesBirths:
            citiesBirths[city] = 0
        
        citiesBirths[city]+=1

        if month not in monthsBirths:
            monthsBirths[month] = 0

        monthsBirths[month]+=1

        totBirths+=1


    print(f"Births per city:\n{citiesBirths}\nBirths per month:{monthsBirths}\nAvg number of births per city: {float(totBirths) / float(len(citiesBirths.keys()))}")

