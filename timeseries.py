import numpy as np


class TimeSeries(object):
    def __init__(self, data = None):
        if data is None:
            data = []
        self.data = data
        self.raw_length = len(data)
        self.sigma_series = [0]*self.raw_length
        self.forecast_length = 0

    def set_data(self, data):
        self.data = data
        self.raw_length = len(data)


class ArimaTS(TimeSeries):
    model_type = 'ARIMA'

    def __init__(self, data = None):
        super(ArimaTS, self).__init__(data)
        self.model = ArimaModel()
        self.epsilon_list = []

    def simulate_series(self, n=100):
        self.raw_length = n
        self.data = []
        epsilon_list = np.random.normal(scale=self.model.sigma, size=n)
        self.epsilon_list = epsilon_list
        for t in range(n):
            yt = self.model.mu
            for i in range(1, self.model.p+1):
                if t >= i:
                    yt += -self.model.phi_list[i]*self.data[t-i]
            for i in range(self.model.q+1):
                if t >= i:
                    yt += self.model.theta_list[i]*epsilon_list[t-i]
            self.data.append(yt)

    def forecast(self, n=10):
        self.forecast_length = n
        for t in range(self.raw_length, self.raw_length+n):
            yt = self.model.mu
            for i in range(1, self.model.p+1):
                if t >= i:
                    yt -= self.model.phi_list[i]*self.data[t-i]
            for i in range(self.model.q+1):
                if t >=i and t-i < self.raw_length:
                    yt += self.model.theta_list[i]*self.epsilon_list[t-i]
            self.data.append(yt)


class ArimaModel(object):
    """
    an arma(p,q) model
    (phi0+phi1*B+phi2*B^2+...+phip*B^p)y_t = mu+(theta0+theta1*B+...+theta_q*B^q)epsilon_t
    epsilon_t~N(0,sigma^2)
    """
    def __init__(self, p=0, q=0):
        self.mu = 0
        self.phi_list = [1]
        self.theta_list = [1]
        self.phi_list.extend([0]*p)
        self.theta_list.extend([0]*q)
        self.sigma = 1
        self.p = p
        self.q = q

    def set_mu(self, mu):
        self.mu = mu

    def set_phi_list(self, phi_list):
        self.phi_list = phi_list
        self.p = len(phi_list)-1

    def set_theta_list(self, theta_list):
        self.theta_list = theta_list
        self.q = len(theta_list)-1

    def set_sigma(self, sigma):
        assert sigma > 0
        self.sigma = sigma


class TransferFunctionTS(TimeSeries):
    model_type = 'transfer function model'

    def __init__(self, data=None, x_series_number=1):
        super(TransferFunctionTS, self).__init__()
        self.x_series_list = []
        for i in range(x_series_number):
            self.x_series_list.append(ArimaTS())
        self.k = x_series_number  # the number of predictor time series
        self.model = TransferFunctionModel(self.k)
        self.n_series = ArimaTS()
        self.v_series_list = [[]*self.k]

    def set_y_series(self, data):
        self.data = data
        self.raw_length = len(data)

    def simulate_series(self, n=100):
        self.raw_length = n
        self.data = []
        self.n_series.simulate_series(n)
        for i in range(self.k):
            self.x_series_list[i].simulate_series(n)

        for i in range(self.k): #generate the k v series
            transfer_function = self.model.transfer_function_list[i]
            for t in range(n):  #generate v0 till v_{n-1}
                vt = 0
                for j in range(1, transfer_function.q+1):
                    if t-j >= 0:
                        vt -= transfer_function.delta_list[j]*self.v_series_list[i][t-j]
                for j in range(transfer_function.p+1):
                    if t-transfer_function.d-j >= 0:
                        vt += transfer_function.omega_list[j]*self.x_series_list[i].data[t-transfer_function.d-j]
                self.v_series_list[i].append(vt)

        for t in range(n): #generate the y series
            yt = self.model.mu + self.n_series.data[i]
            for i in range(self.model.k):
                yt += self.v_series_list[i][t]
            self.data.append(yt)






class TransferFunctionModel(object):
    """
    y_t = mu + V1_t + V2_t +...+ Vk_t + N_t
    V_t = omega(B)/delta(B)*B^d*x_t
    N_t = MA(B)/AR(B)epsilon_t
    """

    def __init__(self, k):
        self.k = k
        self.mu = 0
        self.n_series_model = ArimaModel()
        self.transfer_function_list = []
        for i in range(k):
            self.transfer_function_list.append(TransferFunction())

    def set_mu(self, mu):
        self.mu = mu


class TransferFunction(object):
    def __init__(self, p=1, q=1, d=0):
        self.p = p
        self.q = q
        self.d = d
        self.omega_list = [1]*(self.p+1)
        self.delta_list = [1]*(self.q+1)

    def set_omega_list(self, omega_list):
        self.omega_list = omega_list

    def set_delta_list(self, delta_list):
        self.delta_list = delta_list

    def set_d(self, d):
        self.d = d


def main():
    ts = TransferFunctionTS()
    ts.simulate_series()
    print ts.data
    print ts.model.k
    for i in range(ts.model.k):
        print ts.v_series_list[i]
    print ts.n_series.data

if __name__=='__main__':
    print 'test starts'
    main()
