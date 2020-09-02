
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







