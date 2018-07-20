# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:34:22 2018

@author: Administrator
"""

import os
import pandas as pd
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

input_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/original_data')
output_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data')

train_agg = pd.read_csv(input_dir + '/train_agg.csv',sep='\t', header=0)
train_agg = train_agg.reindex(['USRID'] + list(train_agg.columns[:-1]), axis=1)
train_agg = train_agg.sort_values(by=['USRID'])
train_agg.to_csv(output_dir +'/train_agg.csv',index=False)

train_log = pd.read_csv(input_dir + '/train_log.csv',sep='\t', header=0)
train_log = train_log.sort_values(by=['USRID'])
train_log.to_csv(output_dir +'/train_log.csv',index=False)

train_flg = pd.read_csv(input_dir + '/train_flg.csv',sep='\t', header=0)
train_flg = train_flg.sort_values(by=['USRID'])
train_flg.to_csv(output_dir +'/train_flg.csv',index=False)

test_agg = pd.read_csv(input_dir + '/test_agg.csv',sep='\t', header=0)
test_agg = test_agg.reindex(['USRID'] + list(test_agg.columns[:-1]), axis=1)
test_agg = test_agg.sort_values(by=['USRID'])
test_agg.to_csv(output_dir +'/test_agg.csv',index=False)

test_log = pd.read_csv(input_dir + '/test_log.csv',sep='\t', header=0)
test_log = test_log.sort_values(by=['USRID'])
test_log.to_csv(output_dir +'/test_log.csv',index=False)

time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')