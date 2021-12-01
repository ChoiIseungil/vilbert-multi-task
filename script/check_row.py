import pandas as pd 
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument("--load_csv", type=str) 
args = parser.parse_args()


if __name__ == "__main__" : 
    df = pd.read_csv(args.load_csv) 
    print(f"total rows : {len(df)}")
