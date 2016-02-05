"""
a test on gsa
target: revenue C
predictor: Email_Offer, Online_Ads
"""
from timeseries import *
import cplex
import pandas as pd

a, b = 10000.0, 1000.0
revenue = pd.read_csv('data/Revenue.csv')

ts = TcmTS(data=revenue['Product_C_Revenue'].get_values().tolist()[:-12], x_series_number=2)
ts.x_series_list = [ArimaTS(data=revenue['Email_Offer'].get_values().tolist()),
                    ArimaTS(data=revenue['Online_Ads'].get_values().tolist())]

with open('model/model_C.txt') as f:
    lines = f.readlines()
    coefficients = [float(line.strip().split('\t')[-1]) for line in lines[1:]]

ts.model.set_alpha(coefficients[0])
ts.model.set_beta_list([coefficients[1:6],
                        coefficients[6:11],
                        coefficients[11:]])

ts.forecast(n=12)

c = cplex.Cplex()
c.variables.add(names=['y'+str(i) for i in range(105, 129)])
c.variables.add(names=['x0_'+str(i) for i in range(105, 129)])
c.variables.add(names=['x1_'+str(i) for i in range(105, 129)])

target = {123: 13000, }  # target for y series in format position:target value

for i in range(105, 129):  # set the series minimum change objective
    if target.get(i) is None:
        defined_value = ts.data[i]
        c.objective.set_quadratic_coefficients('y'+str(i), 'y'+str(i), 2.0/(defined_value**2))
        c.objective.set_linear('y'+str(i), -2.0/defined_value)
    else:
        defined_value = target.get(i)
        c.objective.set_quadratic_coefficients('y'+str(i), 'y'+str(i), a**2*2.0/(defined_value**2))
        c.objective.set_linear('y'+str(i), -2.0/defined_value*(a**2))

for i in range(105, 129):
    for j in range(ts.k):
        c.objective.set_quadratic_coefficients('x'+str(j)+'_'+str(i), 'x'+str(j)+'_'+str(i),
                                               2.0/(ts.x_series_list[j].data[i]**2))
        c.objective.set_linear('x'+str(j)+'_'+str(i), -2.0/ts.x_series_list[j].data[i])
#
# for i in range(119, 120):  # set match to target objective
#     c.objective.set_quadratic_coefficients('y'+str(i), 'y'+str(i), a**2*2.0/(51000**2))
#     c.objective.set_linear('y'+str(i), -2.0/51000*(a**2))

for i in range(105, 117):  # add historical data consistent constrain
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=['y'+str(i)], val=[1.0]),
                                       cplex.SparsePair(ind=['x0_'+str(i)], val=[1.0]),
                                       cplex.SparsePair(ind=['x1_'+str(i)], val=[1.0])],
                             senses=['E', 'E', 'E'],
                             rhs=[ts.data[i], ts.x_series_list[0].data[i], ts.x_series_list[1].data[i]])

for i in range(117, 129):  # add predicted y series follow the forecast formula constrain
    ind = ['y'+str(i-j) for j in range(6)]
    ind.extend(['x0_'+str(i-j) for j in range(1, 6)])
    ind.extend(['x1_'+str(i-j) for j in range(1, 6)])
    val = [1.0]
    val.extend(-1.0*temp for temp in coefficients[1:])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)], rhs=[coefficients[0]], senses=['E'])


c.solve()
solution = c.solution.get_values()


def print_latex(x):
    i = 0
    for temp in x:
        print '& '+str(round(temp, 2)),
        i += 1
        if i == 12:
            print
    print
    print '-------------------'

print_latex(solution[:24])
print_latex(solution[24:48])
print_latex(solution[48:])
print '+++++++++++++++++++++++++++++'
print_latex(ts.data[-24:])
print_latex(ts.x_series_list[0].data[-24:])
print_latex(ts.x_series_list[1].data[-24:])
# print ['&'+str(round(temp, 2)) for temp in solution[:22]]
# print [round(temp, 2) for temp in solution[22:44]]
# print [round(temp, 2) for temp in solution[44:]]
# print ts.model.alpha
# print ts.model.beta_list
# print coefficients
# print len(coefficients)

# print [round(temp, 2) for temp in ts.data[-22:]]
# print [round(temp, 2) for temp in ts.x_series_list[0].data[-22:]]
# print [round(temp, 2) for temp in ts.x_series_list[1].data[-22:]]
# print ts.data.__len__()
# print len(ts.x_series_list[0].data)
# print len(ts.x_series_list[1].data)
#

# temp=coefficients[0]+coefficients[1]*ts.data[116]+coefficients[2]*ts.data[115]+coefficients[3]*ts.data[114]+coefficients[4]*ts.data[113]+coefficients[5]*ts.data[112]\
#      +coefficients[6]*ts.x_series_list[0].data[116]+coefficients[7]*ts.x_series_list[0].data[115]+coefficients[8]*ts.x_series_list[0].data[114]+coefficients[9]*ts.x_series_list[0].data[113]+coefficients[10]*ts.x_series_list[0].data[112]\
#      +coefficients[11]*ts.x_series_list[1].data[116]+coefficients[12]*ts.x_series_list[1].data[115]+coefficients[13]*ts.x_series_list[1].data[114]+coefficients[14]*ts.x_series_list[1].data[113]+coefficients[15]*ts.x_series_list[1].data[112]
# print temp
#
