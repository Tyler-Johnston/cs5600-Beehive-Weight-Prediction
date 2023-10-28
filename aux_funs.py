import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math

'''
################################################
# Plotting weight/temperature data from USDA-ARS
# in Tucson, AZ
# bugs to vladimir kulyukin in canvas.
################################################
'''

DATE_INDEX = 0
T_INDEX = 1
W_INDEX = 2

TW_INDEX = [DATE_INDEX, T_INDEX, W_INDEX]

def csv_line_to_ary(ln, col_index):
    ary = []
    for col_index in col_index:
        if col_index == 0:
            ary.append(ln[col_index])
        else:
            ary.append(float(ln[col_index]))
    return ary

def csv_file_to_arys(in_path, col_index):
    recs = []
    with open(in_path, mode='r') as inf:
        lncnt = 0
        for ln in csv.reader(inf):
            if '' not in ln:
                if lncnt == 0:
                    lncnt = 1
                elif lncnt > 0 and len(ln)>0:
                    recs.append(csv_line_to_ary(ln, col_index))
    return recs

def parse_date(dt):
    mdy, hm = dt.split()
    m,d,y = mdy.split('/')
    m,d,y = int(m), int(d), int(y)
    if y == 22:
        y += 2000
    hm    = hm.split(':')
    h, mnt  = int(hm[0]), int(hm[1])
    return m,d,y,h,mnt

def is_date_same(date_1, date_2):
    pd1 = parse_date(date_1)
    pd2 = parse_date(date_2)
    assert len(pd1) == len(pd2) == 5
    for i in range(5):
        if pd1[i] != pd2[i]:
            return False
    return True

def get_month_recs(mon, recs):
    mon_recs = []
    for r in recs:
        pdt = parse_date(r[TP_DATE_INDEX])
        if pdt[0] == mon:
            mon_recs.append(r)
    return mon_recs

def get_tw_recs(recs):
    t_recs, w_recs = [], []
    for d, t, w in recs:
        t_recs.append(t)
        w_recs.append(w)
    return t_recs, w_recs

def cubic_root(x):
    return x**(1/3.0)

def log10(x):
    return math.log(x)

def partition_dataset_into_samples(dataset, num_steps):
	X, y = [], []
	for i in range(len(dataset)):
		# find the end of this sample
		end_ix = i + num_steps
		# check if beyond the dataset
		if end_ix > len(dataset):
			break
		# gather input and output parts of the for the sample
		seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
