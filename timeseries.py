import numpy as np


class TimeSeries(object):
    def __init__(self, data=None):
        if data is None:
            data = []
        self.data = data
        self.raw_length = len(data)
        self.sigma_series = [0]*self.raw_length
        self.forecast_length = 0

    def set_data(self, data):
        self.data = data
        self.raw_length = len(data)

    def current_length(self):
        return self.raw_length+self.forecast_length


class ArimaTS(TimeSeries):
    model_type = 'ARIMA'

    def __init__(self, data=None):
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
        super(TransferFunctionTS, self).__init__(data)
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

        for i in range(self.k):  # generate the k v series
            transfer_function = self.model.transfer_function_list[i]
            for t in range(n):  # generate v0 till v_{n-1}
                vt = 0
                for j in range(1, transfer_function.q+1):
                    if t-j >= 0:
                        vt -= transfer_function.delta_list[j]*self.v_series_list[i][t-j]
                for j in range(transfer_function.p+1):
                    if t-transfer_function.d-j >= 0:
                        vt += transfer_function.omega_list[j]*self.x_series_list[i].data[t-transfer_function.d-j]
                self.v_series_list[i].append(vt)

        for t in range(n):  # generate the y series
            yt = self.model.mu + self.n_series.data[t]
            for i in range(self.model.k):
                yt += self.v_series_list[i][t]
            self.data.append(yt)

    def forecast(self, n=10):
        self.forecast_length = n
        for i in range(self.model.k):
            x_series = self.x_series_list[i]
            x_series.forecast(n=n+self.raw_length-x_series.current_length())
        self.n_series.forecast(n=n+self.raw_length-self.n_series.current_length())
        for i in range(self.k):  # generate the k v series
            transfer_function = self.model.transfer_function_list[i]
            for t in range(self.raw_length, self.raw_length+n):  # generate  till v_{raw_length+n-1}
                vt = 0
                for j in range(1, transfer_function.q+1):
                    if t-j >= 0:
                        vt -= transfer_function.delta_list[j]*self.v_series_list[i][t-j]
                for j in range(transfer_function.p+1):
                    if t-transfer_function.d-j >= 0:
                        vt += transfer_function.omega_list[j]*self.x_series_list[i].data[t-transfer_function.d-j]
                self.v_series_list[i].append(vt)
        for t in range(self.raw_length, self.raw_length+n):  # generate the y series
            yt = self.model.mu + self.n_series.data[t]
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


class TcmTS(TimeSeries):
    model_type = 'TCM'

    def __init__(self, data=None, x_series_number=1):
        super(TcmTS, self).__init__(data)
        self.k = x_series_number
        self.x_series_list = []
        for i in range(self.k):
            self.x_series_list.append(ArimaTS())
        self.model = TcmModel(self.k)

    def set_x_series_list(self, x_series_list):
        for x_series in x_series_list:
            if len(x_series) != self.raw_length:
                print "the length of the xseries does not match with the y series"
        self.k = len(x_series_list)
        self.x_series_list = x_series_list

    def simulate_series(self, n=100):
        self.data = []
        self.raw_length = 100
        for i in range(self.k):
            x_series = self.x_series_list[i]
            x_series.simulate_series(n=n)
        epsilon_series = np.random.normal(0, self.model.sigma, n)
        for t in range(n):
            yt = self.model.alpha + epsilon_series[t]
            beta_list = self.model.beta_list[0]
            for i in range(len(beta_list)):
                if t-i-1 >= 0:
                    yt += beta_list[i]*self.data[t-i-1]
            for i in range(1, self.k):
                beta_list = self.model.beta_list[i]
                x_series = self.x_series_list[i]
                for j in range(beta_list):
                    if t-j-1 >= 0:
                        yt += beta_list[j]*x_series.data[t-j-1]
            self.data.append(yt)

    def forecast(self, n=10):
        self.forecast_length = n
        for x_series in self.x_series_list:  # forecast the x series first
            x_series.forecast(n=self.current_length()-x_series.current_length())
        for t in range(self.raw_length, self.raw_length+n):
            yt = self.model.alpha
            beta_list = self.model.beta_list[0]
            for i in range(len(beta_list)):
                if t-i-1 >= 0:
                    yt += beta_list[i]*self.data[t-i-1]
            for beta_list in self.model.beta_list[1:]:
                for j in range(len(beta_list)):
                    if t-j-1 >= 0:
                        yt += beta_list[j]*x_series.data[t-j-1]
            self.data.append(yt)


class TcmModel(object):
    """
    yt = alpha+beta_{0,1}y_{t-1}+beta_{0,2}y_{t-2}+...+beta_{1,1}x_{1,t-1}+beta_{1,2}x_{1,t-2}+...+beta_{2,1}x_{2,t-1}+....+epsilon_t
    epsilon_t ~ N(0, sigma^2)
    """
    def __init__(self, k):
        self.alpha = 0
        self.sigma = 1
        self.beta_list = [[0] for i in range(k+1)]

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta_list(self, beta_list):
        assert len(beta_list) == len(self.beta_list)
        self.beta_list = beta_list

    def set_sigma(self, sigma):
        self.sigma = sigma


def main():
    ts = TcmTS()
    ts.simulate_series()
    print ts.data
    print ts.k
    for i in range(ts.k):
        print ts.x_series_list[i].data
    # print ts.n_series.data
    print '--------------------'

    ts.forecast()
    print ts.data
    print ts.k
    for i in range(ts.k):
        print ts.x_series_list[i].data
    # print ts.n_series.data
    print '------------------'
    print len(ts.data)
    print len(ts.x_series_list[0].data)
    # print len(ts.n_series.data)
if __name__=='__main__':
    print 'test starts'
    main()
