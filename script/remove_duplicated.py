import pandas as pd 


import argparse 

PATH = '/mnt/nas2/seungil/legacy/'

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--load_csv" , type=str, help="load a csv file to refine") 
    parser.add_argument("--out", default="./out.csv", type=str, help="output file name") 
    args = parser.parse_args() 

    df = pd.read_csv(PATH + args.load_csv) 
    
    #print("--origin csv--") 
    #print(df.tail(20))

    duplicated_TF = df.duplicated(subset=['caption']) 
    #print("---example---") 
    #print(duplicated_TF.head()) 
    #print(duplicated_TF.tail())
    dup = 0
    for item in duplicated_TF : 
        if item : pass # if True , pass 
        else : dup += 1 # if False, counting -> it means check the numnber of remained samples 
    print(f"before refined : {len(df)}") 
    print(f"remained : {dup}") 

    refined_df = df.drop_duplicates(subset=['caption'],keep='first')
    #print("refined_df") 
    #print(refined_df.head())
    #print(refined_df.tail())
    refined_df.to_csv(args.out) 
    print(f"remained pairs : {len(refined_df)}")

