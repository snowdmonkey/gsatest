"""
a test on gsa
target: revenue A
predictor: Online_Ads, TV_Ads, Direct_Mail_Offer
model type: transfer function
"""
from timeseries import *
import cplex
import pandas as pd

a, b = 10000.0, 1000.0
revenue = pd.read_csv('data/Revenue.csv')

ts = TransferFunctionTS(data=revenue['Product_A_Revenue'].get_values().tolist()[:-12], x_series_number=3)
ts.x_series_list = [ArimaTS(data=revenue['Online_Ads'].get_values().tolist()),
                    ArimaTS(data=revenue['TV_Ads'].get_values().tolist()),
                    ArimaTS(data=revenue['Direct_Mail_Offer'].get_values().tolist())]
ts.k = 3

ts.model.D = 1
ts.model.s = 12
ts.model.transfer_function_list[0].D = 1
ts.model.transfer_function_list[0].s = 12
ts.model.transfer_function_list[0].omega_list = [1.858]
ts.model.transfer_function_list[0].delta_list = [1.0]

ts.model.transfer_function_list[1].D = 1
ts.model.transfer_function_list[1].s = 12
ts.model.transfer_function_list[1].omega_list = [0.415]
ts.model.transfer_function_list[1].delta_list = [1.0]

ts.model.transfer_function_list[2].D = 1
ts.model.transfer_function_list[2].s = 12
ts.model.transfer_function_list[2].omega_list = [0.462]
ts.model.transfer_function_list[2].delta_list = [1.0]

ts.n_series.model.p = 0
ts.n_series.model.q = 0

ts.n_series.model.set_phi_list([1])
ts.n_series.model.set_theta_list([1])


ts.calculate_v_series()
ts.calculate_n_series()
ts.n_series.calculate_epsilon_list()

ts.forecast(n=12)


c = cplex.Cplex()
c.variables.add(names=['y'+str(i) for i in range(105, 129)])
for i in range(ts.k):
    c.variables.add(names=['x'+str(i)+'_'+str(j) for j in range(105, 129)])
    c.variables.add(names=['v'+str(i)+'_'+str(j) for j in range(105, 129)])

target = {123: 51000, }  # target for y series in format position:target value

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
        c.objective.set_quadratic_coefficients('v'+str(j)+'_'+str(i), 'v'+str(j)+'_'+str(i),
                                               2.0/(ts.v_series_list[j][i]**2))
        c.objective.set_linear('v'+str(j)+'_'+str(i), -2.0/ts.v_series_list[j][i])
# print ts.data[-10:]

lin_expr, sense, rhs = [], [], []
for i in range(105, 117):  # add historical data consistent constrain
    lin_expr.append(cplex.SparsePair(ind=['y'+str(i)], val=[1.0]))
    sense.append('E')
    rhs.append(ts.data[i])
    for k in range(ts.k):
        lin_expr.append(cplex.SparsePair(ind=['x'+str(k)+'_'+str(i)], val=[1.0]))
        sense.append('E')
        rhs.append(ts.x_series_list[k].data[i])
        lin_expr.append(cplex.SparsePair(ind=['v'+str(k)+'_'+str(i)], val=[1.0]))
        sense.append('E')
        rhs.append(ts.v_series_list[k][i])

c.linear_constraints.add(lin_expr=lin_expr, senses=sense, rhs=rhs)

# for i in range(117, 129):  # add predicted y series follow the forecast formula constrain
#     ind = ['y'+str(i-j) for j in range(ts.model.d+ts.model.D*ts.model.s+1)]
#     ind.extend(['v'+str(k)+'_'+str(i) for k in range(ts.k)])
#     c1, c2 = [1.0, -1.0], [0.0]*(ts.model.s+1)
#     c2[0] = 1.0
#     c2[-1] = -1.0
#     c3 = coefficient_product(c1=c1, c2=c2, p1=ts.model.d, p2=ts.model.D)
#     c3.extend([-1.0]*ts.k)
#     rhs = [ts.n_series.data[i]]
#     c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=c3)], rhs=rhs, senses=['E'])
#     # add delta(B)v_t-omega(B)(1-B)^d(1-B^s)^Dxt=0
#     for k in range(ts.k):
#         ind = ['v'+str(k)+'_'+str(i-j) for j in range(ts.model.transfer_function_list[k].delta_list.__len__())]
#         c1, c2 = [1.0, -1.0], [0.0]*(ts.model.transfer_function_list[k].s+1)
#         c2[0], c2[-1] = 1.0, -1.0
#         c3 = coefficient_product(c1=c1, c2=c2,
#                                  p1=ts.model.transfer_function_list[k].d, p2=ts.model.transfer_function_list[k].D)
#         omega_delta_list = coefficient_product(c1=ts.model.transfer_function_list[k].omega_list, c2=c3)
#         ind.extend(['x'+str(k)+'_'+str(i-j) for j in range(len(omega_delta_list))])
#         val = ts.model.transfer_function_list[k].delta_list[:]
#         val.extend([-temp for temp in omega_delta_list])
#         c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)], rhs=[0.0], senses=['E'])

c.solve()
# solution = c.solution.get_values()
print [c.solution.get_values('y'+str(i)) for i in range(105, 129)]
print [c.solution.get_values('x0_'+str(i)) for i in range(105, 129)]
print [c.solution.get_values('x1_'+str(i)) for i in range(105, 129)]
print [c.solution.get_values('x2_'+str(i)) for i in range(105, 129)]

print [c.solution.get_values('v0_'+str(i)) for i in range(105, 129)]
print [c.solution.get_values('v1_'+str(i)) for i in range(105, 129)]
print [c.solution.get_values('v2_'+str(i)) for i in range(105, 129)]
# print ts.data[100:117]
# print ts.v_series_list[0][100:117]
# print ts.v_series_list[1][100:117]
# print ts.v_series_list[2][100:117]
# print ts.n_series.data[100:117]

