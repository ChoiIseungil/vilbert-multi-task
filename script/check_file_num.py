import os 
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument('--path',type=str) 
args = parser.parse_args() 


#filepath = '../../mnt/nas2/seungil/features/AA7/'
dir_list = os.listdir(args.path) 
print(len(dir_list))
