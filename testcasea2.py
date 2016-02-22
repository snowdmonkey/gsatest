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




# print ts.data[100:117]
# print ts.v_series_list[0][100:117]
# print ts.v_series_list[1][100:117]
# print ts.v_series_list[2][100:117]
# print ts.n_series.data[100:117]