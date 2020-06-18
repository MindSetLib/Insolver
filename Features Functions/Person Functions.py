
import pandas as pd
import numpy as np
import re


# Функция корректировки возраста

def f_age( age ):
    if pd.isnull(age):
        age = 18
    elif age < 18:
        age = 18
    elif age > 70:
        age = 70
    return age


# Функция корректировки стажа

def f_exp( exp ):
    if pd.isnull(exp):
        exp = 0
    elif exp < 0:
        exp = 0
    elif exp > 52:
        exp = 52
    return exp


# Функция определения пола

def f_gender( client_type, client_name, client_gender='not_defined' ):

    if client_type == 'company': #juridic person
        gender_m = 0
        gender_f = 0

    elif client_gender == 'm':
        gender_m = 1
        gender_f = 0

    elif client_gender == 'f':
        gender_m = 0
        gender_f = 1

    else:
        try:
            if len(client_name) < 2:
                gender_m = 0
                gender_f = 0
            elif client_name[-2:].upper() == 'ИЧ':
                gender_m = 1
                gender_f = 0
            elif client_name[-2:].upper() == 'ОГЛЫ':
                gender_m = 1
                gender_f = 0
            elif client_name[-2:].upper() == 'НА':
                gender_m = 0
                gender_f = 1
            elif client_name[-2:].upper() == 'КЫЗЫ':
                gender_m = 0
                gender_f = 1
            else:
                gender_m = 0
                gender_f = 0
        except:
            gender_m = 0
            gender_f = 0

    return gender_m, gender_f


# Функция определения воздаста для м/ж

def f_age_gender( age_gender ):
    age = age_gender[0]
    gender = age_gender[1]
    if gender == 1:
        age_res = age
    else:
        age_res = 18
    return age_res


# Функция формирования признака пол-возраст-стаж

def f_gen_age_exp( gender_m, gender_f, age, exp ):

    if gender_m == 1:
        gender = 1
    elif gender_f == 1:
        gender = 2
    else:
        gender = 0

    if age < 25:
        age = 0
    elif age < 30:
        age = 1
    elif age < 35:
        age = 2
    elif age < 40:
        age = 3
    elif age < 45:
        age = 4
    elif age < 50:
        age = 5
    else:
        age = 6

    if exp < 5:
        exp = 0
    elif exp < 10:
        exp = 1
    elif exp < 15:
        exp = 2
    elif exp < 20:
        exp = 3
    elif exp < 25:
        exp = 4
    elif exp < 30:
        exp = 5
    else:
        exp = 6

    return gender*100 + age*10 + exp


# Функция проверки имени

def f_get_name( client_name ):
    tokenize_re = re.compile(r'[\w\-]+', re.I)
    try:
        name = tokenize_re.findall( str(client_name) ).upper()
        return name
    except Exception:
        return 'ERROR'

def f_check_name( name, names_list ):
    if name.upper() in names_list:
        check = 1
    else:
        check = 0
    return check

def f_check_name_df( df, column_name, column_check_name, names_list ):
    df[column_check_name] = 1 * df[column_name].isin(names_list)
    return df





