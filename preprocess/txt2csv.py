import csv

# Parse the .txt file
file_path = "../dataSet/source_data_set.txt"
output_csv = "../dataSet/sms_data.csv"

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Prepare rows for CSV
rows = []
for line in lines:
    line = line.strip()
    if line:
        label, message = line.split(maxsplit=1)
        rows.append([label, message])

# Write to CSV
with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Label", "Message"])  # Add headers
    writer.writerows(rows)

print(f"Data saved to {output_csv}")