import pandas as pd 
import argparse 
import os 

parser = argparse.ArgumentParser() 
parser.add_argument("--data", type=str,default="FA") 

args = parser.parse_args()

CSVPATH = "../../mnt/nas2/seungil/refined_legacy/"
FEATUREPATH = "../../mnt/nas2/seungil/features/"

if __name__ == "__main__" : 

    csv_path = CSVPATH + args.data + '.csv'
    df = pd.read_csv(csv_path) 
    print(f"\ttotal rows : {len(df)}")
    print('\t',df.index)

    

    feature_path = FEATUREPATH + args.data 
    if os.path.exists(feature_path) : 
        li = os.listdir(feature_path)
        print(f"\ttotal npy features : {len(li)}")
        item_index = sorted(list(map(lambda x: int(x.split('_')[0]), li)))
        # print('\t',item_index)

        # exist_index = df.loc[df['Unnamed: 0'] in item_index].index
        # print(exist_index)
        # print(len(temp)

        ex = []
        gif = 0
        for df_idx, row in df.iterrows(): 
            if row['Unnamed: 0'] in item_index : 
                ex.append(df_idx)
            if row['image url'].split('.')[-1] == 'gif': 
                ex.append(df_idx) 
                gif +=1 
        df_ = df.drop(ex)
        print(f'\t new df rows : {len(df_)}')
        print(f"\tN of gif :{gif}")
        print('\t',df_.head())
        




    else : print("Folder Not Found")


