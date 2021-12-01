# Written by Seungil Lee, Dec 2, 2021

# python script/concat_jsonlines.py --data AA1,AA2,AA3,AA4,AA5,AA6,AA7,FA,GA --out train 

import argparse
import shutil

PATH = '/mnt/nas2/seungil/jsonlines/'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", 
        required = True,
        default = None, 
        type = str, 
        help = "Names of the files to be concatenated (Can read multiple files at once)"
    )
    parser.add_argument(
        "--out", 
        required = True,
        default = None, 
        type = str, 
        help = "Name of merged the file"
    )
    return parser

def main():
    args = get_parser().parse_args()

    files = list()
    for file in args.data.split(','):
        files.append(PATH + file + '.jsonline')
    
    with open(PATH + args.out+'.jsonline','wb') as wfd:
        for f in files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

if __name__ == "__main__":
    main()