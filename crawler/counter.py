# Written by Seungil Lee, Dec 1, 2021

import csv
import os
import glob

PATH = '/mnt/nas2/seungil/refined_legacy/'


def main():
    files = []
    for file in glob.glob(PATH + "/*.csv"): files.append(file)
    
    with open (PATH + 'statistics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file','pairs'])
        for file in files:
            fileName = os.path.basename(file)
            file = open(file,'r')
            reader = csv.reader(file)
            count = len(list(reader)) - 1  # 1 for header row
            writer.writerow([fileName,count])
            print(f"{fileName} has {count} rows")
            file.close()

    f.close()
    
    print("Finished")

if __name__ == '__main__':
    main()