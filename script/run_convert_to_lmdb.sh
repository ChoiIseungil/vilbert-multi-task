#!/bin/bash

python script/convert_to_lmdb.py --data FA,GA,AA0,AA1,AA2,AA3,AA4,AA5,AA6,AA7 --out train
python script/convert_to_lmdb.py --data AA8 --out val
python script/convert_to_lmdb.py --data AA9 --out test
python script/convert_to_lmdb.py --data AA6 --out AA6
python script/convert_to_lmdb.py --data AA7 --out AA7
python script/convert_to_lmdb.py --data AA8 --out AA8
python script/convert_to_lmdb.py --data AA9 --out AA9


