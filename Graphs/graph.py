import csv

all_rows = []

with open("Enc Dec preds.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        all_rows.append(row)

preds = {}
# Print all rows
for i, row in enumerate(all_rows):
    preds[row[0]] = row[1:]

print(preds)