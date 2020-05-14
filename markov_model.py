#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:45:02 2020

@author: ted
"""

import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
#from datetime import datetime
#import scipy
#import itertools
import sympy
import csv


'''
Import data table from MySQL
'''

my_data_names = ['DEFAULTED',
'TENDER_EXCH_OFFER',
 'ANNOUNCED_CALL',
 'ACTIVE_ISSUE',
 'BOND_TYPE',
 'ISIN',
 'ACTION_TYPE',
'EFFECTIVE_DATE',
 'CROSS_DEFAULT',
 'RATING_DECLINE_TRIGGER_PUT',
 'RATING_DECLINE_PROVISION',
 'FUNDED_DEBT_IS',
 'INDEBTEDNESS_IS',
 'FUNDED_DEBT_SUB',
 'INDEBTEDNESS_SUB',
 'CUSIP_NAME',
  'INDUSTRY_GROUP',
   'INDUSTRY_CODE',
   'ESOP',
   'IN_BANKRUPTCY',
   'PARENT_ID',
   'NAICS_CODE',
   'COUNTRY_DOMICILE',
   'SIC_CODE',
   'ISSUE_ID',
   'RATING_TYPE',
   'RATING_DATE',
   'RATING',
   'RATING_STATUS',
   'REASON',
   'RATING_STATUS_DATE',
   'INVESTMENT_GRADE',
   'ISSUER_ID',
   'PROSPECTUS_ISSUER_NAME',
   'ISSUER_CUSIP',
   'ISSUE_CUSIP',
   'ISSUE_NAME',
   'MATURITY',
   'OFFERING_DATE',
   'COMPLETE_CUSIP',
   'N_INDEX' ]

my_data_dtypes = {
   'INDUSTRY_GROUP':float,
   'INDUSTRY_CODE': float,
    'SIC_CODE':float,
   'N_INDEX':int }


my_data_datetime = ['EFFECTIVE_DATE','RATING_DATE',
                    'RATING_STATUS_DATE','MATURITY','OFFERING_DATE']


my_data = pd.io.parsers.read_csv('main_n_index.csv',names=my_data_names,dtype=str)

for col, col_type in my_data_dtypes.items():
    my_data[col] = my_data[col].astype(col_type)

for col in my_data_datetime:
    my_data[col] = pd.to_datetime(my_data[col],yearfirst=True)

print('Data set imported')
'''
Put default information into indexed series
Note - if the bond defaults, and the last rating was NR, replacing that
    with DEFAULT. If the last rating was anything else, adding another
    index value and making that DEFAULT.
'''

my_data['COMPLETE_CUSIP_RATING_TYPE'] = my_data['COMPLETE_CUSIP'] + my_data['RATING_TYPE']
my_data = my_data.drop_duplicates(['COMPLETE_CUSIP_RATING_TYPE','RATING_DATE'])

as_of = pd.to_datetime('20200501',yearfirst=True)

max_n_index = my_data.groupby('COMPLETE_CUSIP_RATING_TYPE').N_INDEX.max()
max_series = pd.merge(my_data,max_n_index,on = ['COMPLETE_CUSIP_RATING_TYPE','N_INDEX'])
max_series = max_series.loc[((max_series.DEFAULTED == 'Y') 
            | ((max_series.DEFAULTED == 'N') & (max_series.MATURITY < as_of)))
            & (((max_series.RATING == 'NR') & (max_series.N_INDEX == 1))
            | (max_series.RATING != 'NR'))]

max_series['N_INDEX'] +=1

max_series.loc[(max_series.DEFAULTED == 'Y'),'RATING'] = 'DEFAULT'
max_series.loc[(max_series.DEFAULTED == 'N'),'RATING'] = 'MATURE'

columns_to_nan = ['RATING_DATE','RATING_STATUS','REASON','RATING_STATUS_DATE',
   'INVESTMENT_GRADE']

for i in columns_to_nan:
    max_series[i] = np.nan
    
my_data = pd.concat([my_data,max_series]).reset_index()


max_n_index = my_data.groupby(['COMPLETE_CUSIP_RATING_TYPE']).N_INDEX.max().reset_index()
my_data = pd.merge(my_data,max_n_index, how = 'left',on = ['COMPLETE_CUSIP_RATING_TYPE'])
my_data = my_data.rename(columns={'N_INDEX_x': 'N_INDEX','N_INDEX_y': 'MAX_N_INDEX'})



my_data = my_data.loc[~(my_data.RATING == 'NR/NR')]
my_data = my_data.loc[~(my_data.DEFAULTED.isnull())]


my_data.loc[(my_data.DEFAULTED == 'Y') & (my_data.N_INDEX == my_data.MAX_N_INDEX) & (my_data.RATING == 'NR'),'RATING'] = 'DEFAULT'
my_data.loc[(my_data.DEFAULTED == 'N') & (my_data.MATURITY < as_of) & (my_data.N_INDEX == my_data.MAX_N_INDEX) & (my_data.RATING == 'NR'),'RATING'] = 'MATURE'





'''
add prior ratings
'''

my_data = my_data.sort_values(by=['COMPLETE_CUSIP_RATING_TYPE','N_INDEX'])

my_data['PRIOR_RATING'] = my_data.RATING.shift(periods=1)
my_data.loc[my_data.N_INDEX ==1, 'PRIOR_RATING'] = np.nan

my_data.loc[(my_data.DEFAULTED == 'N') & (my_data.N_INDEX == my_data.MAX_N_INDEX) & (my_data.MATURITY >= as_of) & (my_data.RATING=='NR'), 'RATING'] = my_data.PRIOR_RATING


my_data = my_data.loc[~((my_data.N_INDEX == 1) & (my_data.MAX_N_INDEX == 1) & (my_data.RATING.isnull()))]



print('Data set configured')

'''
make initial state table, grouped by ratings agency and rating
note probability for each rating agency sums to 1, but table contains all 3 agencies
'''
all_ratings_states = my_data.groupby(['RATING_TYPE','RATING']).COMPLETE_CUSIP.nunique()
initial_ratings_state_count = my_data.where(my_data.N_INDEX == 1).groupby(['RATING_TYPE','RATING']).N_INDEX.sum().reset_index()
initial_ratings_state_total = initial_ratings_state_count.groupby('RATING_TYPE').N_INDEX.sum().reset_index()
initial_ratings_state_count = pd.merge(all_ratings_states,initial_ratings_state_count,on=['RATING_TYPE','RATING'],how='outer')
initial_ratings_state_count = pd.merge(initial_ratings_state_count,initial_ratings_state_total,on='RATING_TYPE',how='outer')
initial_ratings_state_count = initial_ratings_state_count.rename(columns={'N_INDEX_x':'COUNT_BY_RATING','N_INDEX_y':'COUNT_BY_RATING_TYPE'})
isp = initial_ratings_state_count['COUNT_BY_RATING']/initial_ratings_state_count['COUNT_BY_RATING_TYPE']
initial_ratings_state_count['INITIAL_STATE_PROBABILITY'] = isp
initial_state_probability = initial_ratings_state_count

'''
seperate tables by rating agency
'''

def data_separate_by_ratings_agency(input_df,rating_agency_string):
    output_df = input_df.loc[input_df.RATING_TYPE == rating_agency_string]
    return output_df
    
def initial_state_probability_by_rating_agency(input_df,rating_agency_string,file_name_string):
    output_df = input_df.loc[input_df.RATING_TYPE == rating_agency_string]
    result = output_df[['RATING','INITIAL_STATE_PROBABILITY']]
    result.to_csv('{file_name}.csv'.format(file_name = file_name_string),float_format='%.3f')
    return result

def maturity_distribution(input_df,file_name_string):    
    input_df = input_df.loc[input_df.N_INDEX == input_df.MAX_N_INDEX]
    output_df = pd.crosstab(index=input_df.PRIOR_RATING,columns=input_df.RATING,values=input_df.MATURITY,aggfunc = 'max')
    output_df.to_csv('{file_name}.csv'.format(file_name = file_name_string),float_format='%.3f')
    return output_df


fr_data = my_data.iloc[0:0]
fr_data.name = 'fr_data'

mr_data = my_data.iloc[0:0]
mr_data.name = 'mr_data'

spr_data = my_data.iloc[0:0]
spr_data.name = 'spr_data'

fr_data = data_separate_by_ratings_agency(my_data,'FR')
mr_data = data_separate_by_ratings_agency(my_data,'MR')
spr_data = data_separate_by_ratings_agency(my_data,'SPR')

fr_initial_state_probability = initial_state_probability.iloc[0:0]
mr_initial_state_probability = initial_state_probability.iloc[0:0]
spr_initial_state_probability = initial_state_probability.iloc[0:0]

fr_initial_state_probability = initial_state_probability_by_rating_agency(initial_state_probability,'FR','fr_initial_state_probability')
mr_initial_state_probability = initial_state_probability_by_rating_agency(initial_state_probability,'MR','mr_initial_state_probability')
spr_initial_state_probability = initial_state_probability_by_rating_agency(initial_state_probability,'SPR','sprr_initial_state_probability')

fr_maturity_distribution = maturity_distribution(fr_data,'fr_maturity_distribution.csv')
mr_maturity_distribution = maturity_distribution(mr_data,'mr_maturity_distribution.csv')
spr_maturity_distribution = maturity_distribution(spr_data,'spr_maturity_distribution.csv')

print('Initial state distributions created')


'''
make transition matrices for each rating agency
'''


def ratings_values(input_df):
    output_list = input_df['RATING'].unique().tolist()
    return output_list

def transition_probability_matrix(input_df,file_name_string):
    output_df = pd.crosstab(index=input_df.PRIOR_RATING,columns=input_df.RATING,values=input_df.COMPLETE_CUSIP,aggfunc = 'count')
    output_df = output_df.fillna(0)
    output_df_sum = output_df.sum(axis=1)
    output_df = output_df.div(output_df_sum,axis=0)

    if 'DEFAULT' in output_df.index:    
        output_df.iloc[[output_df.index.get_loc('DEFAULT')],:] = float(0)
    else:
        default_df = pd.DataFrame(float(0),columns=output_df.columns,index=['DEFAULT'])
        output_df = output_df.append(default_df)
    output_df.loc['DEFAULT','DEFAULT'] = float(1)

    if 'MATURE' in output_df.index:    
        output_df.iloc[[output_df.index.get_loc('MATURE')],:] = float(0)
    else:
        default_df = pd.DataFrame(float(0),columns=output_df.columns,index=['MATURE'])
        output_df = output_df.append(default_df)
    output_df.loc['MATURE','MATURE'] = float(1)

    output_df = output_df.fillna(0)

    cols = list(output_df.columns.values)
    cols.append(cols.pop(cols.index('DEFAULT')))
    cols.append(cols.pop(cols.index('MATURE')))
    output_df = output_df.reindex(index=cols,columns=cols)

    output_df.to_csv('{file_name}.csv'.format(file_name = file_name_string),float_format='%.3f')
    return output_df

fr_ratings_values = []
mr_ratings_values = []
spr_ratings_values = []

fr_ratings_values = ratings_values(fr_data)
mr_ratings_values = ratings_values(mr_data)
spr_ratings_values = ratings_values(spr_data)

fr_transition_probability_matrix = pd.DataFrame(index=fr_ratings_values,columns=fr_ratings_values)
mr_transition_probability_matrix = pd.DataFrame(index=mr_ratings_values,columns=mr_ratings_values)
spr_transition_probability_matrix = pd.DataFrame(index=spr_ratings_values,columns=spr_ratings_values)


fr_transition_probability_matrix = transition_probability_matrix(fr_data,'fr_transition_probability_matrix')
mr_transition_probability_matrix = transition_probability_matrix(mr_data,'mr_transition_probability_matrix')
spr_transition_probability_matrix = transition_probability_matrix(spr_data,'spr_transition_probability_matrix')

print('Transition probability matrices created')


def absorption_probability(input_df,file_name_string,tol=0.001):
    x = len(input_df.columns)
    q = input_df.iloc[:x-2,:x-2]
    n = np.eye(x-2,x-2)-q 
    n_cond = np.linalg.cond(n, p='fro')
    print('for {df}, the conditioning number is {cond}'.format(df = file_name_string, cond = n_cond))
    r = input_df.iloc[:x-2,x-2:x]
    m = pd.DataFrame(np.zeros((x-2,2)),columns=r.columns,index=r.index)
    for col_name,col in r.iteritems():
        augmented = pd.concat([n, r[col_name]], axis=1)
        augmented_np1 = np.array(augmented.values)
        augmented_rref_matrix = sympy.Matrix(augmented_np1).rref()
        augmented_np2 = np.array(augmented_rref_matrix[0]).astype(np.float64)
        reduced_row_echelon = pd.DataFrame(augmented_np2,columns = augmented.columns, index=augmented.index)
        m[col_name] = reduced_row_echelon[col_name]        
    m.to_csv('{file_name}.csv'.format(file_name = file_name_string),float_format='%.3f')
    return m

fr_absorption_probability = absorption_probability(fr_transition_probability_matrix,'fr_absorption_probability')
mr_absorption_probability = absorption_probability(mr_transition_probability_matrix,'mr_absorption_probability')
spr_absorption_probability = absorption_probability(spr_transition_probability_matrix,'spr_absorption_probability')

print('Absorption probabilities calculated - analytically')



def absorption_probability_simulation(transition_df,file_name_string,max_steps=1000,iter_per_state=1000):
    size = len(transition_df.index)
    row = transition_df.index[:size-2]
    col = transition_df.columns[size-2:size]
    index_list = transition_df.index.to_numpy()
    result = pd.DataFrame(np.zeros([size-2,2]),index=row,columns=col)
    error_df = np.zeros([size-2,1])
    result['RECURRENT'] = error_df
    for x in list(range(len(row))):
        for y in list(range(iter_per_state)):
            i = 0
            while (i < max_steps):       
                state = transition_df.index[x]
                state = np.random.choice(index_list,p=transition_df.iloc[transition_df.index.get_loc(state),:])
                if state == 'DEFAULT':
                    result.iloc[x,result.columns.get_loc('DEFAULT')] +=1
                    break
                if state == 'MATURE':
                    result.iloc[x,result.columns.get_loc('MATURE')] +=1
                    break
                else:
                    i +=1
                    if i == max_steps-1:
                        result.iloc[x,result.columns.get_loc('RECURRENT')] +=1
                        break        
    result = result/ iter_per_state
    result.to_csv('{file_name}.csv'.format(file_name = file_name_string),float_format='%.3f')
    return result


fr_absorption_probability_simulation = absorption_probability_simulation(fr_transition_probability_matrix,'fr_absorption_probability_simulation')
mr_absorption_probability_simulation = absorption_probability_simulation(fr_transition_probability_matrix,'mr_absorption_probability_simulation')
spr_absorption_probability_simulation = absorption_probability_simulation(fr_transition_probability_matrix,'spr_absorption_probability_simulation')
        
print('Absorption probabilities calculated - simulated')

def partition(lst, n): 
    random.shuffle(lst)
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def data_series_hmm(input_df,file_name_string,n_partitions):
    size_columns = input_df.N_INDEX.nunique()
    size_index = input_df.COMPLETE_CUSIP.nunique()
    list_values = list(set(input_df.RATING))
    list_values = [x for x in list_values if str(x) != 'nan']
    list_columns = list(set(input_df.N_INDEX))
    list_index = list(set(input_df.COMPLETE_CUSIP))

    list_values.sort()
    list_columns.sort()
    list_index.sort()

    list_values_keys = list(range(0,len(list_values)))
    dict_values = dict(zip(list_values_keys,list_values))
    inv_dict_values = {v: k for k, v in dict_values.items()}

    output_df = pd.DataFrame(np.zeros((size_index,size_columns)),columns=list_columns,index=list_index)  
    output_df = input_df.pivot(index='COMPLETE_CUSIP', columns='N_INDEX', values = 'RATING')
    output_df = output_df.replace(inv_dict_values)
    output_list = []

    for i in range(len(list_index)):
        x = output_df.iloc[i,:].values
        x_nan = np.isnan(x)
        x_true = ~x_nan
        x = x[x_true]
        x_list = x.tolist()
        for j in range(len(x_list)):
            x_list[j] = int(x_list[j])
        output_list.append(x_list)

    output_list = partition(output_list,n_partitions)
    
    for i in range(n_partitions): 
        with open("{file}_set_{ind}.csv".format(file = file_name_string,ind=i), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(output_list[i])


data_series_hmm(fr_data,'hmm_fr_data',5)
data_series_hmm(mr_data,'hmm_mr_data',5)
data_series_hmm(spr_data,'hmm_spr_data',5)



'''
Find steady state probabilities
'''

'''

def steady_state_vector(transition_matrix):
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(transition_matrix.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    stationary = pd.DataFrame(stationary,index=transition_matrix.columns.tolist())
    return stationary

print(steady_state_vector(fr_transition_probability_matrix))



def steady_state_probabilities(transition_prob_matrix,initial_prob,column_list,conv = 0.0001,max_iter=10000):
    steady_state_0 = pd.DataFrame(index=column_list,columns=column_list)
    steady_state_1 = pd.DataFrame(index=column_list,columns=column_list)
    initial_prob['key']=1
    steady_state_0['key']=1
    steady_state_1['key']=1
    transition_prob_matrix['key']=1
    steady_state_0 = initial_prob.T.dot(transition_prob_matrix.set_index('key'))
    steady_state_1 = steady_state_0.dot(transition_prob_matrix.set_index('key'))
    i = 0
    diff = abs(steady_state_1 - steady_state_0)
    while  ((diff > conv).all(axis=None)) and (i<max_iter):
        steady_state_1 = steady_state_0.dot(transition_prob_matrix.set_index('key'))
        steady_state_0 = steady_state_1
        diff = abs(steady_state_1 - steady_state_0)
        i =+ 1
    print(i)
    print(steady_state_1)
    return steady_state_1



fr_ratings_values = fr_transition_probability_matrix.columns.tolist()

fr_steady_state_probabilities = pd.DataFrame(columns=fr_ratings_values)

fr_initial_state_probability = fr_initial_state_probability.loc[fr_initial_state_probability.RATING.isin(fr_ratings_values)]
mr_initial_state_probability = mr_initial_state_probability.loc[mr_initial_state_probability.RATING.isin(mr_ratings_values)]
spr_initial_state_probability = spr_initial_state_probability.loc[spr_initial_state_probability.RATING.isin(spr_ratings_values)]



fr_steady_state_probabilities = steady_state_probabilities(fr_transition_probability_matrix,fr_initial_state_probability,fr_ratings_values)
'''
    



