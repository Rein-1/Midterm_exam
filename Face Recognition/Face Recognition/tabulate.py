# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:31:07 2023

@author: Rein
"""

import csv
from tabulate import tabulate

# Read test results from a CSV file
test_results = []
with open('recognition.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header
    for row in reader:
        test_results.append(row)

# Check if all images were detected with faces
all_faces_detected = all("face" in result[0] for result in test_results)

# Display table
print("Task 3A: Mood Detection using XYZ Algorithm")
print(tabulate(test_results, headers=header, tablefmt="grid"))

# Calculate accuracy based on face detection
if all_faces_detected:
    print("All images detected a face. 100% accuracy!")
    accuracy = 100
else:
    total = sum(int(result[0].split('_')[-1].split('.')[0]) for result in test_results)
    accuracy = round((total / len(test_results)) * 100, 2)
    print(f"Accuracy: {accuracy}%")
