import cplex


c = cplex.Cplex()
c.variables.add(names=['x', 'y'], lb=[-cplex.infinity]*2)
c.objective.set_quadratic_coefficients('y', 'y', 2.0)
c.objective.set_quadratic_coefficients('x', 'y', 2.0)
c.objective.set_quadratic_coefficients('x', 'x', 2.0)
# print solution
# c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=['x'], val=[1.0])], senses=['E'], rhs=[1.0])

c.solve()

print c.objective.get_quadratic()
# print c.variables.get_lower_bounds()
# print c.variables.get_upper_bounds()
print c.solution.get_values()
print c.solution.get_objective_value()