import os 
import pandas as pd 


import argparse 



if __name__ == "__main__" : 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--load_csv" , type=str, help="load a csv file to refine") 
    parser.add_argument("--out", default="./out.csv", type=str, help="output file name") 
    args = parser.parse_args() 

    df = pd.read_csv(args.load_csv) 
    
    duplicated_TF = df.duplicated(subset=['caption']) 
    dup = 0
    for item in duplicated_TF : 
        if item : pass 
        else : dup += 1 
    print(f"before refined : {len(df)}") 
    print(f"duplicated : {dup}") 

    refined_df = df.drop_duplicates(subset=['caption']) 
    refined_df.to_csv(args.out) 
    print(f"remained pairs : {len(refined_df)}")

