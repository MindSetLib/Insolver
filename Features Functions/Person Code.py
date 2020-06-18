
# Применяем функцию определения пола

df['gender_m'],df['gender_w'] = zip(*df[['client_type','client_name','client_gender']].apply(f_gender, axis=1).to_frame()[0])


# Применяем функцию определения возраста для мужчин и женщин

df['age_m'] = df[['age','gender_m']].apply(f_age_gender, axis=1)
df['age_w'] = df[['age','gender_w']].apply(f_age_gender, axis=1)


# Применяем функцию корректировки минимального возраста

df['age'] = df['age'].apply(f_age)


# Применяем функцию корректировки минимального стажа

df['experience'] = df['experience'].apply(f_exp)


# Проверяем, есть если различие между возрастом и стажем менее 18 лет

df[['age','experience']].loc[(df['age'] - df['experience']) < 18]


# Корректируем, если различие между возрастом и стажем менее 18 лет

df[['age','experience']].loc[(df['age'] - df['experience']) < 18] = [None,None]
df['experience'].loc[(df['age'] - df['experience']) < 18] = df['age'] - 18