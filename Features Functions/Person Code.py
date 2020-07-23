# Применяем функцию корректировки минимального возраста

df['DriverMinAge'] = df['DriverMinAge'].apply(f_DriverAge)


# Применяем функцию корректировки минимального стажа

df['DriverMinExp'] = df['DriverMinExp'].apply(f_DriverExp)


# Проверяем, есть если различие между возрастом и стажем менее 18 лет

df[['DriverMinAge','DriverMinExp']].loc[(df['DriverMinAge'] - df['DriverMinExp']) < 18]


# Корректируем, если различие между возрастом и стажем менее 18 лет

df[['DriverMinAge','DriverMinExp']].loc[(df['DriverMinAge'] - df['DriverMinExp']) < 18] = [None,None]
#df['DriverMinExp'].loc[(df['DriverMinAge'] - df['DriverMinExp']) < 18] = df['DriverMinAge'] - 18
