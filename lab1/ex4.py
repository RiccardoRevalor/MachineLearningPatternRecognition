filename = "ex4_data.txt"

books = {}  #(isbn, (#available_copies, gain_per_sold_copies))
monthYearBooks = {} #(month/year, #books_sold)


with open(filename) as f:
    for line in f:
        field = line.split(" ")
        isbn = field[0]
        flag = field[1]
        ts = field[2].split("/")[1] + "/" + field[2].split("/")[2]
        price_sold = float(field[4])

        if isbn not in books: books[isbn] = [0, 0]

        if flag == "S":
            if books[isbn][0] > 0: 
                books[isbn][0]-=1
            books[isbn][1]+=price_sold

            #update sold books per month/year

            if ts not in monthYearBooks: monthYearBooks[ts] = 0
            monthYearBooks[ts]+=1

        elif flag == "B":
            books[isbn][0]+=1


print(f"Books:\n{books}\nMonth/year boooks:{monthYearBooks}")

