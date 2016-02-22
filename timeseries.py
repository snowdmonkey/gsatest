import numpy as np


def coefficient_product(c1=[1], c2=[1], p1=1, p2=1):
    """
    this function will return the coefficients list of a product of two polynomial
    c1 is the coefficient list of the first polynomial and c2 the second
    the first polynomial is c1[0]+c1[1]*x+c1[2]*x^2+...
    p1 and p2 is the power of the two polynomials, respectively
    """
    c1_input, c2_input = c1[:], c2[:]
    while p1 != 1:
        c1 = coefficient_product(c1, c1_input)
        p1 -= 1
    while p2 != 1:
        c2 = coefficient_product(c2, c2_input)
        p2 -= 1
    c3 = [0]*(len(c1)+len(c2)-1)
    for i in range(len(c1)):
        for j in range(len(c2)):
            c3[i+j] += c1[i]*c2[j]
    return c3


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

    @staticmethod
    def differ_series(series, d=0, D=0, s=1):
        """
        given series x_t, calculate series u_t as u_t = (1-B)^d (1-B^s)^D
        """
        series = series[:]
        if d != 0:
            for i in range(d):
                for j in range(len(series))[::-1]:
                    try:
                        if j-1 >= 0:
                            series[j] = series[j] - series[j-1]
                        else:
                            series[j] = None
                    except:
                        series[j] = None

        if D != 0 and s != 1:
            for i in range(D):
                for j in range(len(series))[::-1]:
                    try:
                        if j-s >= 0:
                            series[j] = series[j] - series[j-s]
                        else:
                            series[j] = None
                    except:
                        series[j] = None
        return series

    @staticmethod
    def filter_series(x_series, omega_list, delta_list):
        """
        calculate and return v_t=omega(B)/delta(B)*x_t
        omega_list is a list contain the coefficient of omega(B)
        delta_list is a list contain the coefficient of delta(B)
        """
        omega1 = reduce(lambda x, y: x+y, omega_list)
        delta1 = reduce(lambda x, y: x+y, delta_list)
        i = 0
        while x_series[i] is None:
            i += 1
        u1 = x_series[i]
        v0 = omega1/delta1*u1
        if delta_list[0] != 1:
            print 'the first coefficient of delta should be 1'
            raise ValueError
        v_series = []
        for t in range(len(x_series)):
            if t < len(omega_list)-1:
                v_series.append(None)
            else:
                vt = 0
                for i in range(len(omega_list)):
                    if x_series[t-i] is None:
                        vt += omega_list[i]*u1
                    else:
                        vt += omega_list[i]*x_series[t-i]
                for i in range(len(delta_list))[1:]:
                    if t-i < 0:
                        vt -= delta_list[i]*v0
                    elif v_series[t-i] is None:
                        vt -= delta_list[i]*v0
                    else:
                        vt -= delta_list[i]*v_series[t-i]
                v_series.append(vt)
        return v_series



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
        if n <= 0:
            return
        self.forecast_length = n
        for t in range(self.raw_length, self.raw_length+n):
            yt = self.model.mu
            for i in range(1, self.model.p+1):
                if t >= i:
                    yt -= self.model.phi_list[i]*self.data[t-i]
            for i in range(self.model.q+1):
                if t >= i and t-i < self.raw_length:
                    yt += self.model.theta_list[i]*self.epsilon_list[t-i]
            self.data.append(yt)

    def calculate_epsilon_list(self):
        u_series = []
        for xt in self.data:
            if xt is None:
                u_series.append(None)
            else:
                u_series.append(xt-self.model.mu)
        self.epsilon_list = self.filter_series(x_series=u_series,
                                               omega_list=self.model.phi_list,
                                               delta_list=self.model.theta_list)


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
        self.v_series_list = [[]]*self.k

    def set_v_series_list(self, v_series_list):
        if len(v_series_list) != self.k:
            print 'v series should be a list of lists with length %s' % self.k
            raise ValueError
        self.v_series_list = v_series_list

    def calculate_v_series(self):
        for i in range(self.k):
            x_series = self.x_series_list[i].data
            transfer_function = self.model.transfer_function_list[i]
            u_series = self.differ_series(x_series, d=transfer_function.d, D=transfer_function.D, s=transfer_function.s)
            v_series = self.filter_series(x_series=u_series,
                                          omega_list=transfer_function.omega_list,
                                          delta_list=transfer_function.delta_list)
            self.v_series_list[i] = v_series

    def calculate_n_series(self):
        n_series = []
        delta_z_series = self.differ_series(series=self.data, d=self.model.d, D=self.model.D, s=self.model.s)
        for i in range(len(self.data)):
            if delta_z_series[i] is None:
                n_series.append(None)
            else:
                nt = delta_z_series[i] - self.model.mu
                for j in range(self.k):
                    if self.v_series_list[j][i] is None:
                        nt = None
                        break
                    else:
                        nt -= self.v_series_list[j][i]
                n_series.append(nt)
        self.n_series.set_data(n_series)

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
                for j in range(1, transfer_function.q+transfer_function.Q*transfer_function.s+1):
                    if t-j >= 0:
                        vt -= transfer_function.delta_list[j]*self.v_series_list[i][t-j]
                for j in range(transfer_function.p+transfer_function.P*transfer_function.s+1):
                    if t-transfer_function.d-j >= 0:
                        vt += transfer_function.omega_list[j]*self.x_series_list[i].data[t-transfer_function.d-j]
                self.v_series_list[i].append(vt)

        for t in range(n):  # generate the y series
            yt = self.model.mu + self.n_series.data[t]
            for i in range(self.model.k):
                yt += self.v_series_list[i][t]
            self.data.append(yt)

    def forecast(self, n=10):
        if n <= 0:
            return
        self.forecast_length = n
        for i in range(self.model.k):
            x_series = self.x_series_list[i]
            x_series.forecast(n=n+self.raw_length-x_series.current_length())
        self.n_series.forecast(n=n+self.raw_length-self.n_series.current_length())
        for i in range(self.k):  # generate the k v series
            transfer_function = self.model.transfer_function_list[i]
            for t in range(self.raw_length, self.raw_length+n):  # generate  till v_{raw_length+n-1}
                vt = 0
                for j in range(1, transfer_function.q+transfer_function.Q*transfer_function.s+1):
                    if t-j >= 0:
                        vt -= transfer_function.delta_list[j]*self.v_series_list[i][t-j]
                for j in range(transfer_function.p+transfer_function.P*transfer_function.s+1):
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

    def __init__(self, k, d=0, D=0, s=1, mu=0):
        self.k = k
        self.mu = mu
        self.d = d
        self.D = D
        self.s = s
        self.n_series_model = ArimaModel()
        self.transfer_function_list = []
        for i in range(k):
            self.transfer_function_list.append(TransferFunction())

    def set_mu(self, mu):
        self.mu = mu


class TransferFunction(object):
    def __init__(self, p=0, q=0, d=0, P=0, Q=0, D=0, s=1, b=0):
        self.p = p
        self.q = q
        self.d = d
        self.P = P
        self.Q = Q
        self.D = D
        self.s = s
        self.b = b
        self.omega_list = [1]*(p+P*s+1)
        self.delta_list = [1]*(q+Q*s+1)

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
            for i in range(1, self.k+1):
                beta_list = self.model.beta_list[i]
                x_series = self.x_series_list[i-1]
                for j in range(len(beta_list)):
                    if t-j-1 >= 0:
                        yt += beta_list[j]*x_series.data[t-j-1]
            self.data.append(yt)

    def forecast(self, n=10):
        if n <=0:
            return
        self.forecast_length = n
        for x_series in self.x_series_list:  # forecast the x series first
            x_series.forecast(n=self.current_length()-x_series.current_length())
        for t in range(self.raw_length, self.raw_length+n):
            yt = self.model.alpha
            beta_list = self.model.beta_list[0]
            for i in range(len(beta_list)):
                if t-i-1 >= 0:
                    yt += beta_list[i]*self.data[t-i-1]
            for i in range(1, self.k+1):
                beta_list = self.model.beta_list[i]
                x_series = self.x_series_list[i-1]
                for j in range(len(beta_list)):
                    if t-j-1 >= 0:
                        yt += beta_list[j]*(x_series.data[t-j-1])
            self.data.append(yt)


class TcmModel(object):
    """
    yt = alpha+beta_{0,1}y_{t-1}+beta_{0,2}y_{t-2}+...+beta_{1,1}x_{1,t-1}+beta_{1,2}x_{1,t-2}+...
         +beta_{2,1}x_{2,t-1}+....+epsilon_t
    epsilon_t ~ N(0, sigma^2)
    """
    def __init__(self, k):
        self.alpha = 0
        self.sigma = 1
        self.beta_list = [[0]]*(k+1)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta_list(self, beta_list):
        assert len(beta_list) == len(self.beta_list)
        self.beta_list = beta_list

    def set_sigma(self, sigma):
        self.sigma = sigma


def main():
   ts = TransferFunctionTS()
   print coefficient_product(c1=[1, -1], p1=3, c2=[1, 2], p2=20)
if __name__ == '__main__':
    print 'test starts'
    main()
