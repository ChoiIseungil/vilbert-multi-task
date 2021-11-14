#Written by Seungil Lee, 12 Nov 2021
#crawled AA0.csv -> AA0.jsonl

import os
import numpy as np
import pandas as pd
import jsonlines as jsonlines
import argparse

DATAPATH = "/mnt/nas2/seungil/result_legacy/"
SAVEPATH = "/mnt/nas2/seungil/jsonlines/"
FEATUREPATH = "/mnt/nas2/seungil/old_dataset/"
 

def write_json(fname):
    df = pd.read_csv(DATAPATH+fname+'.csv')
    # df = df[['image url', column]].groupby('image url')[column].apply(list).reset_index(name = column)
    df.drop_duplicates(subset=['image url'], inplace=True)

    flist = []
    success, fail = 0,0
    for i, row in df.iterrows():
        temp = {}
        temp['image_id'] = row['title'].replace(".","") + '_' + str(i)
        temp['caption'] = row['caption']
        temp['context'] = row['title'] + " " + row['paragraph'] + " " + row['contexts'] #concatenate
        if os.path.exists(FEATUREPATH + fname + '_features/' + temp['image_id'] + '.npy'): 
            flist.append(temp)
            success += 1
        else: 
            fail += 1
            continue

    print(f"{success} lines generated, {fail} files doesn't exist")

    with jsonlines.open(SAVEPATH + fname + '.jsonline', 'w') as writer:
        writer.write_all(flist)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
      "-fname",
      type=str,
      required = True,
      help="name of csv file that will be loaded. e.g. AA0"
    )

    args = parser.parse_args()
    
    write_json(args.fname)

if __name__ == "__main__":
    main()