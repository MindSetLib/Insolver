
import pandas as pd
import numpy as np


# Функция корректировки мощности

def f_power ( power ):
    if pd.isnull(power):
        power = None
    elif power <= 10:
        power = None
    elif power > 500:
        power = None
    else:
        power = round( power/10, 0 )
    return power


# Функция корректировки возраста авто

def f_vehicle_age ( age ):
    if pd.isnull(age):
        age = None
    elif age < 0:
        age = None
    elif age > 20:
        age = 20
    return age


# 