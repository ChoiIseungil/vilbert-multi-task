#Written by Seungil Lee, 12 Nov 2021
#Updated by Beomjin Seo, 23 Dec 2021 

# e.g.
# python script/json_generator.py --fname FA-GA-AA1-AA2-AA3-AA4-AA5-AA6-AA7 --out train.jsonlines >>> total pairs : 65382
# python script/json_generator.py --fname AA8 --out val.jsonlines >>> total pairs : 3527
# python script/json_generator.py --fname AA9 --out test.jsonlines >>> total pairs : 9675

import os
import pandas as pd
import jsonlines as jsonlines
import argparse

import re 


CSVPATH = "/mnt/nas2/seungil/refined_legacy/"
SAVEPATH = "/mnt/nas2/seungil/jsonlines/"
FEATUREPATH = "/mnt/nas2/seungil/features/"


def write_json(fname):

    df = pd.read_csv(CSVPATH + fname+'.csv')
    flist = []
    success, fail = 0,0

    feature_dir = os.listdir(FEATUREPATH + fname + '/')

    for item in feature_dir : 
        # print(f"FILE : {item}")
        feature_name = item.split('.npy')[0]
        title = re.sub(r'[0-9]+_','',feature_name)
        index = int(feature_name.split('_')[0])
        
        item_from_csv = df[df['Unnamed: 0'] == index]
        title_from_csv = item_from_csv['title'].item().replace(".","")
        
        if title_from_csv == title : 
            temp = {}
            temp['image_id'] = feature_name 
            temp['caption'] = item_from_csv['caption'].item()
            temp['context'] = item_from_csv['contexts'].item()
            flist.append(temp)
            success +=1 

        else : 
            fail +=1
            print("error sample : ", item)

    print(f"{fname} jsonlines are extracted.")

    print(f"N of features : {len(feature_dir)}")
    print(f"N of csv row : {len(df)}")
    print(f"success : {success}\tfail : {fail}\n")
    return flist 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
      "--fname",
      type=str,
      required = True,
      help="name of csv file that will be used to make a specific jsonline file. e.g. FA-GA-AA1-AA2-AA3-AA4-AA5-AA6-AA7"
    )
    parser.add_argument(
      "--out",
      type=str,
      default="train.jsonlines",
      help="outfile name"
    )

    args = parser.parse_args()

    fname_list = args.fname.split("-")
    print(f"data list : {fname_list}\n")

    flist = []
    for fname in fname_list : 
        flist.extend(write_json(fname))

    with jsonlines.open(SAVEPATH + args.out , 'w') as writer:
        writer.write_all(flist)

    print(f"total pairs : {len(flist)}")
    print(f"{args.out} is saved")    
    
if __name__ == "__main__":
    main()