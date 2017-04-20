import random
import sys
import os
import csv

with open("emDict.csv", "r",encoding="utf8") as csvsample:
    csv_reader = csv.reader(csvsample,delimiter = ';')
    emDict = list(csv_reader)

    for row in emDict:
       print(row)

csvsample.close()

