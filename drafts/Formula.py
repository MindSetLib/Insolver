list_of_variables = ['cf_0_200',
 'ac_0_200',
 'cf_200_400',
 'ac_200_400',
 'cf_400',
 'ac_400',
 'cf_court',
 'ac_court',
 'cf_ish',
 'ac_pvu_client',
 'ac_pvu_return',
 'cf_bi',
 'ac_bi']

eq1 = "cf_0_200*ac_0_200 + cf_200_400*ac_200_400 + cf_400*ac_400 + cf_court*ac_court + cf_ish * (ac_pvu_client + ac_pvu_return) + cf_bi*ac_bi"


dict_variables = {i:1 for i in list_of_variables}

for key, value in dict_variables.items():
    print(key,value)
    

formula = sympify(eq1)
result = float(formula.subs(dict_variables).evalf())


formula = sympify(eq1)
for i, j in itertools.product(range(50), range(50)):
    dict_variables['cf_0_200'] = i
    dict_variables['ac_0_200'] = j
    result = float(formula.subs(dict_variables).evalf())