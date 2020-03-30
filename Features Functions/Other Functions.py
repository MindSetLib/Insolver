
import pandas as pd
import numpy as np


# Функция полиномайзер

def f_polynomizer( data, column, n=2 ):
    for i in range(2,n+1):
        data[ column + '_' + str(i) ] = data[column]**i
    return data


# Функция создания dummy-переменных
# Есть функция pd.getdummies(df,columns)

def f_dummy( data, column ):
    spr_dummy = data[column].unique()
    for i in spr_dummy:
        data[column+'_'+str(i)] = ( 1 * ( data[column]==i ) ).astype(float)
    return data


# Функция корректировки суммы убытков в предыдущем полисе

def f_claim_sum_pol_prev_calibration( data, column='claim_sum_pol_prev', n=10 ):
    claimsum = data[column][data[column]>0].sort_values(column,ascending=True)
    l = int( len(claimsum)/10 )
    a = []
    for i in range(n):
        a.append( claimsum.iloc[l*i] )
    return a

def f_claim_sum_pol_prev( claim_sum_pol_prev, claim_sum_pol_prev_calibration, n=10 ):
    if pd.isnull( claim_sum_pol_prev )
        claim_sum_pol_prev = 0
    else:
        i=0
        while i < n:
            if claim_sum_pol_prev >= int(claim_sum_pol_prev_calibration[i]):
                i+=1
                continue
            else:
                break
        claim_sum_pol_prev = i
    return claim_sum_pol_prev




