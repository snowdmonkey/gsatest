import cplex


c = cplex.Cplex()
c.variables.add(names=['y'+str(i) for i in range(107, 120)])
c.variables.add(names=['x0_'+str(i) for i in range(107, 120)])
c.variables.add(names=['x1_'+str(i) for i in range(107, 120)])

data = range(200)

a = 10000

for i in range(107, 119):  # set the series minimum change objective
    c.objective.set_quadratic_coefficients('y'+str(i), 'y'+str(i), 2.0/(data[i]**2))
    c.objective.set_linear('y'+str(i), -2.0/data[i])
    for j in range(2):
        c.objective.set_quadratic_coefficients('x'+str(j)+'_'+str(i), 'x'+str(j)+'_'+str(i),
                                               2.0/(data[i]**2))
        c.objective.set_linear('x'+str(j)+'_'+str(i), -2.0/data[i])

for i in range(119, 120):  # set match to target objective
    c.objective.set_quadratic_coefficients('y'+str(i), 'y'+str(i), a**2.0*2/(51000**2))
    c.objective.set_linear('y'+str(i), -2.0/51000*(a**2))


print data[117]

# # c.variables.add(names=['x', 'y'])
# c.objective.set_quadratic_coefficients('y107', 'y107', 2.0/(117**2))
# c.objective.set_linear('y107', -2.0/117)


c.solve()
solution = c.solution.get_values()
print solution[:13]
print solution[13:26]
print solution[26:]
# print solution