# coding:utf-8

'''
__name__ ='test'
__Audthor__ = 'zzds'
__version__ ='0.1.0'
'''
import data_pro as dp
import numpy as np
import pandas as pd
import os, sys
import copy
import re
import xgboost as xgb

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append(os.getcwd())


#write correct borrow_test
def rewrite():
    borrow_test = open(r'.\data\test\borrow_test.txt','r')
    new_borrow = open(r'.\data\test\new_borrow_test.txt','ab+')
    borrow_data = []
    for line in borrow_test.readlines():
        if line.count('"')%2 == 0:
            borrow_data.append(line)
    borrow_test.close()
    for datas in borrow_data:
        new_borrow.write(datas) 
    new_borrow.close()
#rewrite()
    
#导入数据 test 数据
borrow_test = pd.read_csv(r'.\data\test\new_borrow_test.txt',header=None,names=['stu_id','borrow_date','book_name','book_type'],parse_dates = ['borrow_date'],encoding="utf-8")
dorm_test = pd.read_csv(r'.\data\test\dorm_test.txt',header=None,names=['stu_id','act_time','direction'],parse_dates = ['act_time'])
card_test = pd.read_csv(r'.\data\test\card_test.txt',header=None,names=['stu_id','cost_type','cost_pos','cost_reason','cost_time','cost_amount','rest_amount'],parse_dates = ['cost_time'])
library_test = pd.read_csv(r'.\data\test\library_test.txt',header=None,names=['stu_id','gate','time'],parse_dates = ['time'])
score_test = pd.read_csv(r'.\data\test\score_test.txt',header=None,names=['stu_id','department_id','rank'])
student = pd.read_csv(r'.\data\test\studentID_test.txt',header=None,names=['stu_id']) #test表
##去重
card_test = card_test[card_test.cost_amount>0] #沃日
borrow_test.drop_duplicates(inplace = True)
dorm_test.drop_duplicates(inplace = True)
library_test.drop_duplicates(inplace = True)
card_test.drop_duplicates(inplace = True)
################################################数据预处理 （主要是card_test）##################################################
card_test['year'] = card_test.cost_time.dt.year
card_test['weekday'] = card_test.cost_time.dt.weekday+1
card_test['weekofyear'] = card_test.cost_time.dt.weekofyear
card_test['date'] = card_test.cost_time.dt.date
card_test['weekend'] = card_test.weekday.apply(lambda x :'weekend' if x>5 else 'not_weekend')

#去除从不使用卡的记录
user_card_last_time  = card_test.groupby(['stu_id'])['cost_time'].max() - card_test.groupby(['stu_id'])['cost_time'].min()
user_card_date_counts = card_test.groupby(['stu_id'])['date'].unique().apply(lambda x:len(x)).to_frame().rename(columns = {'date':'count_date'})
user_card_date_counts.reset_index(inplace = True)
user_card_last_time = user_card_last_time.to_frame().reset_index()
user_card_last_time.cost_time = user_card_last_time.cost_time.dt.days
user_card_last_time = user_card_last_time.rename(columns = {'cost_time':'timediff'})
card_test[~card_test.stu_id.isin(user_card_last_time.stu_id)]
stu_year = card_test.groupby(['stu_id'])['cost_time'].max().dt.year.to_frame().rename(columns = {'cost_time':'subsidy_year'}).reset_index()
stu_year.subsidy_year = stu_year.subsidy_year.apply(lambda x: 2014 if x==2015 else 2013)
student = student[student.stu_id.isin(user_card_last_time.stu_id)]#去除训练集中从来没有出现过得数据
student = student.merge(stu_year,on = 'stu_id')
#查找交互记录少于N天的人的助学金信息
student[user_card_last_time.timediff<10].count()
student[(user_card_date_counts.count_date<20).values].count()



#borrow data
def cope_borrow_data(borrow_test):
    pattern = r'[A-Z0-9]+'
    #regex = re.compile(pattern,flags = re.IGNORECASE)
    borrow_test.book_type =  borrow_test.book_type.str.split(' ',expand = True).iloc[:,0].fillna('NAN').str.findall(pattern,flags = re.IGNORECASE)
    #find false data
    #book_type =  borrow_test.book_type.str.split(' ',expand = True).iloc[:,0].fillna('NAN')
    #data = []
    #for i,i_type in enumerate(book_type):
    #    data.append(regex.findall(i_type)[0])
    #borrow_test.iloc[i]
    borrow_test['book_type'] = borrow_test['book_type'].apply(lambda x:x[0])
    return
cope_borrow_data(borrow_test)

def groupMerge(target,data,col):
    index1 = data.index.levels[0]
    for i_index in index1:
        tmp = data.ix[i_index].reset_index().rename(columns = {col:col+str(i_index)})
        target = target.merge(tmp,on = 'stu_id',how = 'left')
    return target
    
#特征
cost_sum = card_test.groupby(['stu_id'],as_index = False)[['cost_amount']].sum().rename(columns = {'cost_amount':'cost_amount_sum'})
rest_mean = card_test.groupby(['stu_id'],as_index = False)[['rest_amount']].mean().rename(columns = {'rest_amount':'rest_amount_mean'})
groupcard_cost_mean = card_test.groupby(['cost_reason','stu_id'])[['cost_amount']].mean().rename(columns = {'cost_amount':'cost_amount_mean'})
groupcard_cost_type_sum = card_test.groupby(['cost_type','stu_id'])[['cost_amount']].sum().rename(columns = {'cost_amount':'cost_amount_sum'})
groupcard_cost_type_mean = card_test.groupby(['cost_type','stu_id'])[['cost_amount']].mean().rename(columns = {'cost_amount':'cost_amount_mean'})
groupcard_cost_type_count = card_test.groupby(['cost_type','stu_id'])[['cost_amount']].count().rename(columns = {'cost_amount':'cost_type_count'})
groupcard_cost_reason_count = card_test.groupby(['cost_reason','stu_id'])[['cost_amount']].count().rename(columns = {'cost_amount':'cost_reason_count'})
groupcard = card_test.groupby(['cost_reason','stu_id'])[['cost_amount']].sum().rename(columns = {'cost_amount':'cost_amount_sum'})
groupborrow = borrow_test.groupby(['book_type','stu_id'])[['book_name']].count().rename(columns = {'book_name':'borrow_count'}).sum(level = 1).reset_index()
grouplibrary = library_test.groupby('stu_id',as_index = False)[['time']].count().rename(columns = {'time':'libraryCount'})
groupdorm = dorm_test.groupby(['direction','stu_id'])[['act_time']].count().rename(columns = {'act_time':'dormCount'})
groupbyweekend = card_test.groupby(['weekend','stu_id'])[['cost_amount']].sum().rename(columns = {'cost_amount':'cost_amount_sum'})
groupbyweekday = card_test.groupby(['weekday','stu_id'])[['cost_amount']].sum().rename(columns = {'cost_amount':'cost_amount_sum'})
        
#关联features (简单特征)
test_data = student.copy()
test_data = test_data.merge(score_test,on = 'stu_id',how = 'left').fillna(0)
test_data = groupMerge(test_data,groupcard,'cost_amount_sum').fillna(0)
test_data = test_data.merge(groupborrow,on = 'stu_id',how = 'left').fillna(0)
test_data = test_data.merge(grouplibrary,on = 'stu_id',how = 'left').fillna(0)
test_data = groupMerge(test_data,groupdorm,'dormCount').fillna(0)
test_data = groupMerge(test_data,groupcard_cost_mean,'cost_amount_mean').fillna(0)
test_data = test_data.merge(cost_sum,on = 'stu_id',how = 'left').fillna(0)
test_data = test_data.merge(rest_mean,on = 'stu_id',how = 'left').fillna(0)
test_data = groupMerge(test_data,groupcard_cost_type_sum,'cost_amount_sum').fillna(0)
test_data = groupMerge(test_data,groupcard_cost_type_mean,'cost_amount_mean').fillna(0)
test_data = groupMerge(test_data,groupcard_cost_type_count,'cost_type_count').fillna(0)
test_data = groupMerge(test_data,groupcard_cost_reason_count,'cost_reason_count').fillna(0)
test_data = test_data.merge(user_card_last_time,on = 'stu_id').fillna(0)
test_data = test_data.merge(user_card_date_counts,on = 'stu_id').fillna(0)
test_data = groupMerge(test_data,groupbyweekend,'cost_amount_sum').fillna(0)
test_data = groupMerge(test_data,groupbyweekday,'cost_amount_sum').fillna(0)



#def find_seq(data):
#    count = data.value_counts().to_frame()
#    count.sort_values(by = 'cluster',inplace = True,ascending = False)
#    count['order'] = np.arange(4)
#    data = data.replace(dict(zip(count.index,count.order)))
#    return data.values
    
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer   #标准化
#X = test_data.values
#Scaler = MinMaxScaler()
#X = Scaler.fit_transform(X)
#kmeans = KMeans(n_clusters=4)
#kmeans.fit(X)
#test_data['cluster'] = kmeans.labels_
#ans = find_seq(test_data['cluster'])
#test_data['ans'] = 0
#test_data['ans'][test_data.index] = ans
#test_data[['stu_id','ans']].replace({1:1000,2:1500,3:2000}).astype('int64').to_csv(r'.\ans\anwser_1118_1.csv',index = False,header = True)

#加载模型并且预测
test_data1 = test_data[[
'department_id',
'cost_amount_mean卡片销户',
'cost_amount_sumnot_weekend',
'cost_amount_mean卡补办',
'cost_amount_sum卡补办',
'cost_amount_mean圈存转账',
'cost_amount_sumweekend',
'cost_reason_count食堂',
'cost_type_countPOS消费',
'cost_amount_mean洗衣房',
'cost_amount_sum食堂',
'cost_reason_count文印中心',
'cost_reason_count超市',
'cost_reason_count开水',
'cost_reason_count图书馆',
'cost_type_count卡充值',
'cost_amount_meanPOS消费',
'cost_amount_sum开水',
'cost_reason_count洗衣房',
'cost_reason_count校车',
'cost_amount_sum7',
'cost_amount_sum4',
'dormCount1',
'cost_amount_sum超市',
'cost_amount_mean校车',
'rest_amount_mean',
'dormCount0',
'cost_amount_sum校车',
'cost_amount_mean开水',
'cost_amount_sum6',
'cost_amount_sum3',
'cost_amount_sum2',
'cost_amount_sum5',
'count_date',
'cost_amount_sum卡充值',
'cost_amount_mean食堂',
'cost_amount_sum教务处',
'cost_reason_count淋浴',
'cost_amount_sum1',
'cost_amount_mean教务处',
'cost_amount_sum图书馆',
'rank',
'cost_amount_sum淋浴',
'cost_amount_mean卡充值',
'cost_amount_sum洗衣房',
'cost_amount_mean淋浴',
'cost_amount_mean超市',
'libraryCount',
'borrow_count',
'cost_amount_sum文印中心',
'cost_amount_mean图书馆',
'cost_amount_mean文印中心',
'stu_id'
]]
clf = xgb.Booster({'nthread':4}) #init model
clf.load_model(r'.\model\myxgb_5.m') # load data
Dtest = xgb.DMatrix(test_data1)
predict = clf.predict(Dtest)  #studentid,subsidy
print sum(predict)
student['subsidy'] = predict
student['subsidy'] = student['subsidy'].replace({1:1000,2:1500,3:2000}).astype('int64')
studentall = pd.read_csv(r'.\data\test\studentID_test.txt',header=None,names=['stu_id']) #test表
studentall = studentall.merge(student,on='stu_id',how = 'left').fillna(0).astype('int64')
studentall.rename(columns = {'stu_id':'studentid'},inplace = True)
studentall[['studentid','subsidy']].to_csv(r'.\ans\anwser_1120_1.csv',index = False,header = True)
