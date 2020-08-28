import pandas as pd


# Функция создания dummy-переменных
# Есть функция pd.get_dummies(df, columns)
def f_dummy(data, column):
    spr_dummy = data[column].unique()
    for i in spr_dummy:
        data[f'{column}_{i}'] = (1 * (data[column] == i)).astype(float)
    return data


# Функция корректировки суммы убытков в предыдущем полисе
def f_claim_sum_pol_prev_calibration(data, column='claim_sum_pol_prev', n=10):
    claim_sum = data[column][data[column] > 0].sort_values(column, ascending=True)
    length = int(len(claim_sum)/10)
    a = []
    for i in range(n):
        a.append(claim_sum.iloc[length * i])
    return a


def f_claim_sum_pol_prev(claim_sum_pol_prev, claim_sum_pol_prev_calibration, n=10):
    if pd.isnull(claim_sum_pol_prev):
        claim_sum_pol_prev = 0
    else:
        i = 0
        while i < n:
            if claim_sum_pol_prev >= int(claim_sum_pol_prev_calibration[i]):
                i += 1
                continue
            else:
                break
        claim_sum_pol_prev = i
    return claim_sum_pol_prev
