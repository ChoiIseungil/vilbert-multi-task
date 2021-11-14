#Written by Seungil Lee, 12 Nov 2021
#crawled AA0.csv -> AA0.jsonl

import os
import numpy as np
import pandas as pd
import jsonlines as jsonlines
import argparse

import re 


CSVPATH = "/mnt/nas2/seungil/result_legacy/"
SAVEPATH = "/mnt/nas2/seungil/jsonlines/"
FEATUREPATH = "/mnt/nas2/seungil/old_dataset/"
 

def write_json(fname):
    df = pd.read_csv(CSVPATH+fname+'.csv')

    # df = df[['image url', column]].groupby('image url')[column].apply(list).reset_index(name = column)
    # df.duplicated(['image url'])
    # df.drop_duplicates(subset=['image url'], inplace=True)

    flist = []
    success, fail = 0,0

    feature_dir = os.listdir(FEATUREPATH + fname + '_features/')

    for item in feature_dir : 
        feature_name = item.split('.npy')[0]
        title = re.sub(r'_[0-9]+','',feature_name)
        index = int(feature_name.split('_')[-1])

        if df.iloc[index]['title'].replace(".","") == title : 
            temp = {}
            temp['image_id'] = feature_name 
            temp['caption'] = df.iloc[index]['caption']
            temp['context'] = df.iloc[index]['contexts']
            flist.append(temp)
            success +=1 

        else : 
            fail +=1
            print("error sample : ", item)

    # print(f"flist0 : {flist[0]}")
            
    with jsonlines.open(SAVEPATH + fname + '.jsonline', 'w') as writer:
        writer.write_all(flist)
    
    print(f"N of features : {len(feature_dir)}")
    print(f"N of csv row : {len(df)}")
    print(f"success : {success}\nfail : {fail}")

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