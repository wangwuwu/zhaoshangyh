# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:58:58 2018

@author: ASUS
"""

import os
import gc
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
end_time = '2018-03-31 23:59:59'
end_time = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
begin_time = '2018-03-01 00:00:00'
begin_time = time.mktime(time.strptime(begin_time, "%Y-%m-%d %H:%M:%S"))

temp_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/log_temp.csv')
data_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/preprocessed_data/')
train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1DF-data/train_and_test/test.csv')

def flg_agg_log(is_test):
    if is_test:
        train_agg = pd.read_csv(data_path + 'train_agg.csv',nrows=500)
        test_agg = pd.read_csv(data_path + 'test_agg.csv',nrows=500)
        train_flg = pd.read_csv(data_path + 'train_flg.csv',nrows=500)
        train_log = pd.read_csv(data_path + 'train_log.csv',nrows=500)
        test_log = pd.read_csv(data_path + 'test_log.csv',nrows=500)
    else:
        train_agg = pd.read_csv(data_path + 'train_agg.csv')
        test_agg = pd.read_csv(data_path + 'test_agg.csv')
        train_flg = pd.read_csv(data_path + 'train_flg.csv')
        train_log = pd.read_csv(data_path + 'train_log.csv')
        test_log = pd.read_csv(data_path + 'test_log.csv')
    
    test_flg = pd.DataFrame(test_agg['USRID'])
    test_flg['FLAG'] = -1
    flg = pd.concat([train_flg,test_flg],copy=False)
    agg = pd.concat([train_agg,test_agg],copy=False)
    log = pd.concat([train_log,test_log],copy=False)
    return flg,agg,log

def expand_log(is_load_log,is_save_log,log):
    if is_load_log:
        log = pd.read_csv(temp_path)
    else:
        log['time'] = log['OCC_TIM'].apply(lambda x:time.strptime(x, "%Y-%m-%d %H:%M:%S")).values
        log['total_second'] = log['time'].apply(lambda x:time.mktime(x))-begin_time
        log = log.sort_values(['USRID','total_second'])
        log['total_minute'] = (log['total_second']/60).apply(np.int)
        log['total_hour'] = (log['total_minute']/60).apply(np.int)
        log['day'] = log['time'].apply(lambda x:x.tm_mday)
        log['hour'] = log['time'].apply(lambda x:x.tm_hour)
        log['weekday'] = log['time'].apply(lambda x:x.tm_wday) + 1
        log['next_second'] = log.groupby(['USRID'])['total_second'].diff(-1).apply(np.abs)
        log[['EVT_1','EVT_2','EVT_3']] = log['EVT_LBL'].str.split('-',expand=True)
        if is_save_log:
            log.to_csv('temp_path',index=False)
    return log

def gen_feat_1(log):
    print('generating feat set 1...')
    #ONE-HOT EVT 统计
    log_USRID = log['USRID'].values
    log_EVT = log[['EVT_1','EVT_2','EVT_3']]
    del log
    gc.collect()
    EVT_count = pd.DataFrame(agg['USRID'])
    for i in range(1,4):
        EVT = 'EVT_'+str(i)
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = label_encoder.fit_transform(log_EVT[EVT])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        del integer_encoded
        gc.collect()
        EVT_list = [EVT+'_'+str(x) for x in range(onehot_encoded.shape[1])]
        log_df = pd.DataFrame(onehot_encoded,columns=EVT_list,dtype=int)
        del onehot_encoded
        gc.collect()
        log_df['USRID'] = log_USRID.reshape(len(log_USRID), 1)
        for evt in EVT_list:
            EVT_count_temp = log_df[log_df[evt]==1].groupby(['USRID'],as_index=False)[evt].agg({evt+'_count':'count'})
            EVT_count = pd.merge(EVT_count,EVT_count_temp,on=['USRID'],how='left',copy=False) 
        del log_df
        gc.collect()
    return EVT_count

def gen_feat_2():
    print('generating feat set 2...')
    #行为时间特征
    day_info = log.groupby(['USRID'],as_index=False)['day'].agg({'day_mean':np.mean,'day_std':np.std,'day_min':np.min,'day_max':np.max,'day_skew':lambda x: pd.Series.skew(x),'day_kurt':lambda x: pd.Series.kurt(x)}) 
    day_info['day_rest'] = (day_info['day_max']-31).apply(np.abs)
    day_info['duration'] = (day_info['day_max']-day_info['day_min'])+1
    hour_info = log.groupby(['USRID'],as_index=False)['total_hour'].agg({'hour_mean':np.mean,'hour_std':np.std,'hour_min':np.min,'hour_max':np.max,'hour_skew':lambda x: pd.Series.skew(x),'hour_kurt':lambda x: pd.Series.kurt(x)}) 
    hour_info['hour_rest'] = (hour_info['hour_max']-np.int(end_time/3600)).apply(np.abs)
    return day_info,hour_info

def gen_feat_3():
    print('generating feat set 3...')
    #行为相隔时间特征
    next_time = log.groupby(['USRID'],as_index=False)['next_second'].agg({'next_second_mean':np.mean,'next_second_std':np.std,'next_second_min':np.min,'next_second_max':np.max,'next_second_skew':lambda x: pd.Series.skew(x),'next_second_kurt':lambda x: pd.Series.kurt(x)}) 
    next_time['next_minute_mean'] = next_time['next_second_mean']/60
    next_time['next_minute_std'] = next_time['next_second_std']/60
    next_time['next_minute_min'] = next_time['next_second_min']/60
    next_time['next_minute_max'] = next_time['next_second_max']/60
    next_time['next_minute_skew'] = next_time['next_second_skew']/60
    next_time['next_minute_kurt'] = next_time['next_second_kurt']/60
    next_time['next_hour_mean'] = next_time['next_minute_mean']/60
    next_time['next_hour_std'] = next_time['next_minute_std']/60
    next_time['next_hour_min'] = next_time['next_minute_min']/60
    next_time['next_hour_max'] = next_time['next_minute_max']/60
    next_time['next_hour_kurt'] = next_time['next_minute_kurt']/60
    next_time['next_hour_skew'] = next_time['next_minute_mean']/60
    next_time['next_day_mean'] = next_time['next_hour_mean']/24
    next_time['next_day_std'] = next_time['next_hour_std']/24
    next_time['next_day_min'] = next_time['next_hour_min']/24
    next_time['next_day_max'] = next_time['next_hour_max']/24
    next_time['next_day_skew'] = next_time['next_hour_skew']/24
    next_time['next_day_kurt'] = next_time['next_hour_kurt']/24
    day_unique = log.groupby(['USRID','day']).agg({'USRID':'mean','day':'mean'}).rename({'day': 'day_unique'},axis=1)
    day_unique['next_day_unique'] = day_unique.groupby(['USRID'])['day_unique'].diff(-1).apply(np.abs)
    next_day_unique = day_unique.groupby(['USRID'],as_index=False)['next_day_unique'].agg({'next_day_unique_mean':np.mean,'next_day_unique_std':np.std,'next_day_unique_min':np.min,'next_day_unique_max':np.max,'next_day_unique_skew':lambda x: pd.Series.skew(x),'next_day_unique_kurt':lambda x: pd.Series.kurt(x)}) 
    next_time = pd.merge(next_time,next_day_unique,on=['USRID'],how='left',copy=False)
    return next_time

def gen_feat_4():
    print('generating feat set 4...')
    #滑窗 窗口内行为次数
    EVT_count_31to31 = log.loc[log['day']>=31].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_31to31':'count'})
    EVT_count_30to31 = log.loc[log['day']>=30].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_30to31':'count'})
    EVT_count_29to31 = log.loc[log['day']>=29].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_29to31':'count'})
    EVT_count_25to31 = log.loc[log['day']>=25].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_25to31':'count'})
    EVT_count_18to31 = log.loc[log['day']>=18].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_18to31':'count'})
    EVT_count_11to31 = log.loc[log['day']>=11].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_11to31':'count'})
    EVT_count_4to31 = log.loc[log['day']>=4].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_4to31':'count'})
    EVT_count_1to31 = log.loc[log['day']>=1].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to31':'count'})
    EVT_count_1to3 = log.loc[(log['day']>=1)&(log['day']<=3)].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to3':'count'})
    EVT_count_1to7 = log.loc[(log['day']>=1)&(log['day']<=7)].groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_count_1to7':'count'})
    EVT_count_slide = pd.merge(EVT_count_1to31,EVT_count_31to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_30to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_29to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_25to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_18to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_11to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_4to31,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_1to3,on=['USRID'],how='left',copy=False)
    EVT_count_slide = pd.merge(EVT_count_slide,EVT_count_1to7,on=['USRID'],how='left',copy=False)
    return EVT_count_slide

def gen_feat_5():
    print('generating feat set 5...')
    #按天统计行为次数特征
    everyday_count_temp = log.groupby(['USRID','day'],as_index=False)['day'].agg({'everyday_count':'count'})
    everyday_count = everyday_count_temp.groupby(['USRID'],as_index=False)['everyday_count'].agg({'everyday_count_mean':np.mean,'everyday_count_std':np.std,'everyday_count_min':np.min,'everyday_count_max':np.max,'everyday_count_skew':lambda x: pd.Series.skew(x),'everyday_count_kurt':lambda x: pd.Series.kurt(x)})
    #最多行为在哪天
    most_EVT_day = everyday_count_temp.groupby(['USRID'],as_index=False).apply(lambda x: x[x.everyday_count==x.everyday_count.max()]).rename({'day':'most_EVT_day'},axis=1).drop('everyday_count',axis=1)
    most_EVT_day['most_EVT_day_sub_end_day'] = (most_EVT_day['most_EVT_day']-31).apply(np.abs)
    return everyday_count,most_EVT_day

def gen_feat_6():
    print('generating feat set 6...')
    #按小时统计行为次数特征
    everyhour_count_temp = log.groupby(['USRID','total_hour'],as_index=False)['total_hour'].agg({'everyhour_count':'count'})
    everyhour_count = everyhour_count_temp.groupby(['USRID'],as_index=False)['everyhour_count'].agg({'everyhour_count_mean':np.mean,'everyhour_count_std':np.std,'everyhour_count_min':np.min,'everyhour_count_max':np.max,'everyhour_count_skew':lambda x: pd.Series.skew(x),'everyhour_count_kurt':lambda x: pd.Series.kurt(x)})
    return everyhour_count

#最多行为在周几------

def gen_feat_7():
    print('generating feat set 7...')
    #周几数量和占比------
    weekday_temp = log.groupby(['USRID','weekday'],as_index=False)['weekday'].agg({'weekday_count':'count'})
    MON_count = weekday_temp[weekday_temp.weekday==1].drop('weekday',axis=1).rename({'weekday_count':'MON_count'},axis=1)
    TUES_count = weekday_temp[weekday_temp.weekday==2].drop('weekday',axis=1).rename({'weekday_count':'TUES_count'},axis=1)
    WED_count = weekday_temp[weekday_temp.weekday==3].drop('weekday',axis=1).rename({'weekday_count':'WED_count'},axis=1)
    THUR_count = weekday_temp[weekday_temp.weekday==4].drop('weekday',axis=1).rename({'weekday_count':'THUR_count'},axis=1)
    FRI_count = weekday_temp[weekday_temp.weekday==5].drop('weekday',axis=1).rename({'weekday_count':'FRI_count'},axis=1)
    SAT_count = weekday_temp[weekday_temp.weekday==6].drop('weekday',axis=1).rename({'weekday_count':'SAT_count'},axis=1)
    SUN_count = weekday_temp[weekday_temp.weekday==7].drop('weekday',axis=1).rename({'weekday_count':'SUN_count'},axis=1)
    weekday_count = pd.merge(EVT_count_slide[['USRID','EVT_count_1to31']],MON_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,TUES_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,WED_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,THUR_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,FRI_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,SAT_count,on=['USRID'],how='left',copy=False)
    weekday_count = pd.merge(weekday_count,SUN_count,on=['USRID'],how='left',copy=False)
    weekday_count['MON_pct'] = weekday_count['MON_count']/weekday_count['EVT_count_1to31']
    weekday_count['TUES_pct'] = weekday_count['TUES_count']/weekday_count['EVT_count_1to31']
    weekday_count['WED_pct'] = weekday_count['WED_count']/weekday_count['EVT_count_1to31']
    weekday_count['THUR_pct'] = weekday_count['THUR_count']/weekday_count['EVT_count_1to31']
    weekday_count['FRI_pct'] = weekday_count['FRI_count']/weekday_count['EVT_count_1to31']
    weekday_count['SAT_pct'] = weekday_count['SAT_count']/weekday_count['EVT_count_1to31']
    weekday_count['SUN_pct'] = weekday_count['SUN_count']/weekday_count['EVT_count_1to31']
    weekday_count = weekday_count.drop('EVT_count_1to31',axis=1)
    return weekday_count

def gen_feat_8():
    print('generating feat set 8...')
    #各TYP数量和占比
    TYP_temp = log.groupby(['USRID','TCH_TYP'],as_index=False)['TCH_TYP'].agg({'TYP_count':'count'})
    TYP0_count = TYP_temp[TYP_temp.TCH_TYP==0].drop('TCH_TYP',axis=1).rename({'TYP_count':'TYP0_count'},axis=1)
    #TYP1_count = TYP_temp[TYP_temp.TCH_TYP==1].drop('TCH_TYP',axis=1).rename({'TYP_count':'TYP1_count'},axis=1)
    TYP_count = pd.merge(EVT_count_slide[['USRID','EVT_count_1to31']],TYP0_count,on=['USRID'],how='left',copy=False)
#    TYP_count = pd.merge(TYP_count,TYP1_count,on=['USRID'],how='left',copy=False)
    TYP_count['TYP0_pct'] = TYP_count['TYP0_count']/TYP_count['EVT_count_1to31']
    #TYP_count['TYP1_pct'] = TYP_count['TYP1_count']/TYP_count['EVT_count_1to31']
    TYP_count = TYP_count.drop('EVT_count_1to31',axis=1)
    return TYP_count
    
flg,agg,log = flg_agg_log(is_test=0)
log = expand_log(is_load_log=1,is_save_log=0,log=log)
EVT_count = gen_feat_1(log=log)
day_info,hour_info = gen_feat_2()
next_time = gen_feat_3()
EVT_count_slide = gen_feat_4()
everyday_count,most_EVT_day = gen_feat_5()
everyhour_count = gen_feat_6()
weekday_count = gen_feat_7()
TYP_count = gen_feat_8()
#MERGE
data = pd.merge(flg,agg,on=['USRID'],how='left',copy=False)
data = pd.merge(data,day_info,on=['USRID'],how='left',copy=False)
data = pd.merge(data,hour_info,on=['USRID'],how='left',copy=False)
data = pd.merge(data,next_time,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count_slide,on=['USRID'],how='left',copy=False)
data = pd.merge(data,everyday_count,on=['USRID'],how='left',copy=False)
data = pd.merge(data,everyhour_count,on=['USRID'],how='left',copy=False)
#data = pd.merge(data,most_EVT_day,on=['USRID'],how='left',copy=False)
data = pd.merge(data,TYP_count,on=['USRID'],how='left',copy=False)
data = pd.merge(data,weekday_count,on=['USRID'],how='left',copy=False)
data = pd.merge(data,EVT_count,on=['USRID'],how='left',copy=False)
data['avg_EVT'] = data['EVT_count_1to31']/data['duration']
#data = data.fillna(0) #特定列fillna，有些列不需要

train_data = data[data['FLAG']!=-1]
test_data = data[data['FLAG']==-1]
train_data.to_csv(train_path,index=False)
test_data.to_csv(test_path,index=False)

time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
