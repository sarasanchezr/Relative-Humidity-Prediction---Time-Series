from scipy.stats import chi2
from statsmodels.tsa.exponential_smoothing import ets
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time
from scipy import signal
import math

import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams
from statsmodels.tsa.stattools import kpss




#####################################################
#                 ADF and KPSS Test                 #
#####################################################
# From Lab 1
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] <= 0.05:
        print("Reject the null hypothesis(Ho). Series has no unit root and is stationary")
    else:
        print("Time series has a unit root, si it is non stationary ")
        #Weak evidence against null hypothesis,

def KPSS_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


    
def cal_rolling_mean_var(data, start, end):    
    rolling_mean = []    
    rolling_var  = []    
    for i in range(0, len(data)):
        rolling_mean.append(np.mean(data[:i] ))
        rolling_var.append(np.var(data[:i]))
        
    dates = pd.date_range(start=start, end=end, periods=len(data))
    print(len(dates )  )
    # subplot 2 x 1 
    fig = plt.figure()
    plt.subplot(211)

    plt.title('Rolling mean', fontsize=10)
    plt.plot(dates, rolling_mean,'b-',label = 'Mean')
    plt.xlabel('date')
    plt.subplots_adjust(hspace=0.5)
    plt.legend(loc='center right')

    
    plt.subplot(212)
    
    plt.title('Rolling Variance ', fontsize=10)
    plt.plot(dates, rolling_var, 'y-', label='Variance')
    plt.xlabel('date')
    plt.legend(loc='center right')
    #plt.xticks(rotation=45)
    plt.subplots_adjust(hspace=0.5)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()
    return rolling_mean, rolling_var



def differencing(data, interval):
   diff = []
   for i in range(interval, len(data)):
      value = data[i] - data[i - interval]
      diff.append(value)
   return np.array(diff)


def differencing_l(data, interval):
   diff = []
   for i in range(interval, len(data),interval):
      value = data[i] - data[i - interval]
      diff.append(value)
   return np.array(diff)


#####################################################
#          BASE MODEL FUNCTIONS  - Lab 5            #
#####################################################


# Function Average Method
def average_method(train, h_step):
    return [np.round(np.mean(train), 8) for i in range(0, h_step)]

# Function Naive Method
def naive_method(train, h_step):
    return [train[-1] for i in range(0, h_step)]

# Function Drift Method
def drift_method(train, h_step):
    predicted_values = []
    for i in range(0, h_step):
        predicted_value = train[-1] + (i + 1) * ((train[-1] - train[0]) / (len(train) - 1))
        predicted_values.append(round(predicted_value, 3))
    return predicted_values


# holt_linear_winter_method
def holt_linear_winter(train_data, test_data, seasonal_period: int, trend="mul", seasonal="mul",
                               trend_damped=False):
    holt_winter = ets.ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal,
                                           seasonal_periods=seasonal_period, damped=trend_damped).fit()
    holt_winter_forecast = list(holt_winter.forecast(len(test_data)))
    return holt_winter_forecast


def plot_multiline_chart_pandas_using_index(list_of_dataframes, y_axis_common_data, list_of_label, list_of_color,
                                            x_label, y_label, title_of_plot, rotate_xticks=False):
    for i, df in enumerate(list_of_dataframes):
        df[y_axis_common_data].plot(label=list_of_label[i], color=list_of_color[i])
    plt.title(title_of_plot, fontsize=22)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(loc='best', fontsize=20)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.figure(figsize=(16, 10))
    plt.show()


#####################################################
#        Correlation Coeffficient - Lab 2           #
#####################################################

def correlation_coefficent_cal(x, y):
    mean_x_lista = []
    mean_y_lista = []
    diff = []
    x = np.array(x)
    y = np.array(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    m = 0

    for m in range(len(x)):
        #numerator
        x_diffe = x[m] - mean_x #diff between variable and mean
        y_diffe = y[m] - mean_y #diff between variable and mean
        xy = x_diffe * y_diffe
        diff.append(xy)
        # to calculate the denominator
        mean_x_lista.append((x_diffe ** 2)) #the square of the diff between variable and mean
        mean_y_lista.append((y_diffe ** 2))  #the square of the diff between variable and mean

        sum_diff = sum(diff)
        Denominator_x = math.sqrt(sum(mean_x_lista)) # square root of the prevous step
        Denominator_y = math.sqrt(sum(mean_y_lista))
        Denominator = Denominator_x * Denominator_y

    if Denominator == 0: # to consider the cases with denominator = 0
            r = 0
    else:
            r = round(sum_diff / Denominator, 3)

    return r





#####################################################
#        FUNCTIONS LAB 08                        #
#####################################################



def get_num_den(theta, n_a, n_b):
    # get max order from n_a, n_b
    order = max( [ n_a, n_b ]) + 1
    # getting NUM and DEN from THETA
    den = np.array( [1.] + list( theta[:n_a] ) )  # first n_a elements
    num = np.array( [1.] + list( theta[n_a:] ) )  # last n_b elements
    # fill with zeros
    num = np.pad(num, (0, order - n_b - 1), 'constant')
    den = np.pad(den, (0, order - n_a - 1), 'constant')
    return num, den


def compute_e_theta(y, theta, n_a, n_b):
    num, den = get_num_den(theta, n_a, n_b)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e.reshape(len(y), )
    


def compute_derivative(y, theta, n_a, n_b, delta):
    X = []
    e_theta = compute_e_theta(y, theta, n_a, n_b)
    for i in range(n_a + n_b):
        theta_x = theta.copy()
        theta_x[i] += delta    # increment delta
        
        e_theta_d = compute_e_theta(y, theta_x, n_a, n_b)
        
        x_i = ( e_theta - e_theta_d ) / delta
        X.append(x_i)
        
    X = np.column_stack(X)
    A = np.dot(X.T, X)
    g = np.dot(X.T, e_theta)
    return e_theta, X, A, g


def step_01(y, theta, e_theta, n_a, n_b, delta ):
    # Compute E_theta
    e_theta = compute_e_theta(y, theta, n_a, n_b)
    # copmute SSE of E_theta
    SSE_theta = np.dot(e_theta, e_theta)
    # compute derivative
    _, X, A, g =  compute_derivative(y, theta, n_a, n_b, delta)
    return SSE_theta, X, A, g


def step_02(y, theta, A, g, n_a, n_b, mu=1e-2):
    delta_theta = np.dot( np.linalg.inv( A + mu * np.eye( n_a + n_b) ) , g)
    theta_new = theta + delta_theta
    e_theta_new = compute_e_theta(y, theta_new, n_a, n_b)
    SSE_theta_new = np.dot(e_theta_new, e_theta_new)
    return theta_new, delta_theta, e_theta_new, SSE_theta_new




def LM_algoritmh( y, n_a, n_b, num_iter, delta, flip_val=False):
    T = len(y)
    # delta = 1e-6
    max_iter = 100
    mu_max = 1.
    mu = 0.01
    
    SSE_list = []
    
    # STEP 0: Initialization
    theta = np.zeros(shape = (n_a + n_b), dtype=np.float64)
    e_theta = y         # due to theta is equal to zero
    
    # STEP 1: Compute SSE of e
    SSE_theta = np.dot(e_theta, e_theta)
    _, X, A, g =  compute_derivative(y, theta, n_a, n_b, delta)
    
    SSE_list.append(SSE_theta)
    # SSE_theta, X, A, g = step_01(y, theta, e_theta, n_a, n_b, delta )
    
    
    # STEP 2: 
    # delta_theta = np.dot( np.linalg.inv( A + mu * np.eye(n_a + n_b) ) , g)
    # theta_new = theta + delta_theta
    # e_theta_new = compute_e_theta(y, theta_new, n_a, n_b)
    # SSE_theta_new = np.dot(e_theta_new, e_theta_new)
    
    theta_new, delta_theta, e_theta_new, SSE_theta_new = step_02(y, theta, A, g, n_a, n_b, mu)

    ro2 = 0
    cov_theta = np.zeros(shape = A.shape, dtype=np.float64)
    i = 0
    while i < num_iter:
        
        if i < max_iter:
            
            
            # print('Iteration ', i, ':' , SSE_theta, SSE_theta_new)
            
            
            # ===== The error is decreasing ===== #
            if SSE_theta_new < SSE_theta:
                if np.linalg.norm( delta_theta ) < 1e-3:
                    theta = theta_new
                    ro2 = SSE_theta_new / (T - (n_a+n_b))
                    cov_theta = ro2 * np.linalg.inv( A )
                    # print('END: ----> ||DELTA_THETA|| = ', np.linalg.norm( delta_theta ) )
                    
                    break
                else:
                    # print('Old theta: ', theta)
                    # print('New theta: ', theta_new)
                    # print('Old_mu : ', mu )
                    theta = theta_new
                    mu = mu / 10
                    
                    SSE_list.append( SSE_theta )
                    # print('New_mu : ', mu )
                    # print( 'step 2' )
            
            # ===== The error is increasing ===== #        
            while SSE_theta_new >= SSE_theta:
                mu = mu * 10
                # print('SSE_theta new > SSE_theta ', i)
                if mu > mu_max:
                    print('Error Grather than :', mu, mu_max)
                    break
                # RETURN TO STEP 2
                theta_new, delta_theta, e_theta_new, SSE_theta_new = step_02(y, theta, A, g, n_a, n_b, mu)
            
            # ===== Increae Iteration ===== #
            i +=1
            
            if i > max_iter:
                print('Result: #ITERATION grather than MAX_ITER')
                break
            
            # theta = theta_new
            

            # RETURN TO STEP 1 
            # RETURN TO STEP 2
            SSE_theta, X, A, g = step_01(y, theta, e_theta, n_a, n_b, delta )
            theta_new, delta_theta, e_theta_new, SSE_theta_new = step_02(y, theta, A, g, n_a, n_b, mu)
            # print('XXXX')
            # SSE_theta = SSE_theta_new


    if flip_val:
        theta = np.array( list(-1 * theta[:n_a]) + list(theta[n_a:]))
        
    return theta, ro2, cov_theta, SSE_list

        


def zero_pole_plot(num, den,filename=None):

    b = num
    a = den
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    
    # Adding title 
    plt.title('Zero Pole Map')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k








def forecast(y, theta, n_a, n_b):
    # Just for ARMA(2,2), for others lower than ( 2,2 ) filling with zeros
    y_train = y.copy()
    
    order = 3
    den = np.array( [1.] + list( theta[:n_a] ) )  # first n_a elements
    num = np.array( [1.] + list( theta[n_a:] ) )  # last n_b elements
    # fill with zeros
    num = np.pad(num, (0, order - n_b - 1), 'constant')
    den = np.pad(den, (0, order - n_a - 1), 'constant')
    

    theta = np.array(list(den[1:]) + list(num[1:]))
    
    
    y_hat_t_1 = []
    for i in range(0, len(y_train)):
        if i == 0 :
            y_hat_t_1.append( -theta[0]*y_train[i] + theta[2]*y_train[i] )
        elif i == 1:
            y_hat_t_1.append( -theta[0]*y_train[i] - theta[1]*y_train[i-1] + theta[2]*(y_train[i]-y_hat_t_1[i-1]) + theta[3]*y_train[i-1])
        else:
            y_hat_t_1.append( -theta[0]*y_train[i] - theta[1]*y_train[i-1] + theta[2]*(y_train[i]-y_hat_t_1[i-1]) + theta[3]*(y_train[i-1] - y_hat_t_1[i-2] ) )
    
    return np.array( y_hat_t_1 )





def plot_predicted(real, predicted, x_axes, title=''):
    plt.figure()
    plt.title(title)
    plt.plot(list(x_axes), real, label='Real')
    plt.plot(list(x_axes), predicted, label='Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()
    




def plot_acfx(autocorrelation, title_of_plot, x_axis_label="Lags", y_axis_label="Magnitude"):
    # make a symmetric version of autocorrelation using slicing
    symmetric_autocorrelation = autocorrelation[:0:-1] + autocorrelation
    x_positional_values = [i * -1 for i in range(0, len(autocorrelation))][:0:-1] + [i for i in
                                                                                     range(0, len(autocorrelation))]
    # plot the symmetric version using stem
    rcParams['figure.figsize'] = 16, 10
    plt.stem(x_positional_values, symmetric_autocorrelation, use_line_collection=True)
    plt.xlabel(x_axis_label, fontsize=20)
    plt.ylabel(y_axis_label, fontsize=20)
    plt.title(title_of_plot, fontsize=22)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.figure(figsize=(16, 10))
    plt.show()


# Autocorrelation Function:
def calc_acf(y,lags):
    y = np.array(y)
    y_mean = y.mean()
    y_len = len(y)
    s_r_acf = []
    for lag in range(0,lags):
        num=0 #numerator
        den=0 #denominator
        for t in range(lag,y_len):
            num += (y[t]-y_mean) * (y[t-lag]-y_mean)
        for t in range(0,y_len):
            den += (y[t]-y_mean)**2
        r_acf=num/den
        s_r_acf.append(r_acf)
    return s_r_acf # sum of acf



# ACF and PACF Funtion to plot:
def ACF_PACF_Plot(y, title1,title2, nlags=50):
    # calculate ACF and PACF
    acf = calc_acf(y, 20)
    pacf = sm.tsa.stattools.pacf(y, nlags=nlags) #### lags
    
    # Built figure to plot
    print(pacf)
    fig = plt.figure()
    plt.subplot(211)
    plot_acf(y, ax = plt.gca(), lags=nlags) #### lags
    plt.subplot(211).set_title(title1, fontsize=15)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.subplot(212)
    plot_pacf(y, ax = plt.gca(), lags=nlags) #### lags
    plt.subplot(212).set_title(title2, fontsize=15)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
###################################

########## For the GPAC
def auto_corr(y,k):
    T = len(y)
    y_mean = np.mean(y)
    res_num = 0
    res_den = 0
    for t in range(k,T):
        res_num += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0,T):
        res_den += (y[t] - y_mean)**2

    res = res_num/res_den
    return res


def auto_corr_cal(y,k):
    res = []
    for t in range(0,k):
        result = auto_corr(y,t)
        res.append(result)
    return res



def create_gpac_table(j_scope, k_scope, ry, precision=5):
    # initialization: gpac table
    gpac_table = np.zeros(shape=(j_scope, k_scope), dtype=np.float64)
    # iretarion: loop over j_scope
    for j in range(j_scope):

        #************** MAX DENOMINATOR INDICES ***************
        # creating the largest denominator
        max_denominator_indices = np.zeros(shape = (k_scope, k_scope), dtype=np.int64)
        for k in range(k_scope):
            max_denominator_indices[:, k] = np.arange(j - k, j + k_scope - k)
        #************************* END ************************* 
        
        for k in range(1, k_scope + 1):
            #************ APT DENOMINATOR INDICES *****************
            #  slicing largest denominator by K
            apt_denominator_indices = max_denominator_indices[-k:, -k:]
            #************************* END ************************* 
         
            #******************* NUMERATOR INDICES *******************
            # replacing denominator's last columnn with index starting from j+1 upto k times for numerator
            
            numerator_indices = np.copy(apt_denominator_indices)
            # take the 0,0 indexed value and create a range of values from (indexed_value+1, indexed_value+k)
            indexed_value = numerator_indices[0, 0]
            y_matrix = np.arange(indexed_value + 1, indexed_value + k + 1)
            # replacing the last column with this new value
            numerator_indices[:, -1] = y_matrix
            
            #************************** END ************************** 

            #******************* COMPUTE PHI VALUE *******************
            # taking the absolute values
            denominator_indices = np.abs(apt_denominator_indices)
            numerator_indices = np.abs(numerator_indices)
            # replacing the indices with ACF's values
            denominator = np.take(ry, denominator_indices)
            numerator   = np.take(ry, numerator_indices) 
            # takeing the determinant
            denominator_det = np.round(np.linalg.det(denominator), precision)
            numerator_det = np.round(np.linalg.det(numerator), precision)
            # divide it and get the value of phi
            phi_value = np.round(np.divide(numerator_det, denominator_det), precision)

            #************************** END ************************** 
            gpac_table[j, k - 1] = phi_value


    gpac_table_pd = pd.DataFrame(data=gpac_table, columns=[k for k in range(1, k_scope + 1)])

    return gpac_table_pd


# def plot_heatmap(corr_df, title, xticks=None, yticks=None, x_axis_rotation=0, annotation=True):
#     sns.heatmap(corr_df, annot=annotation)
#     plt.title(title)
#     if xticks is not None:
#         plt.xticks([i for i in range(len(xticks))], xticks, rotation=x_axis_rotation)
#     if yticks is not None:
#         plt.yticks([i for i in range(len(yticks))], yticks)
#     plt.show()



def plot_heatmap(corr_df, title, xticks=None, yticks=None, x_axis_rotation=0, annotation=True):
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(corr_df, annot=annotation, cmap="mako",linewidth=0.3, linecolor='w') #,fontsize = 15 
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 15)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 15)
    plt.title(title, fontdict={'fontsize': 20}, pad=12)
    if xticks is not None:
        plt.xticks([i for i in range(len(xticks))], xticks, rotation=x_axis_rotation)
    if yticks is not None:
        plt.yticks([i for i in range(len(yticks))], yticks)
    plt.show()


import statsmodels.api as sm

def chi_square_test(Q, lags, n_a, n_b, alpha=0.01):
    dof = lags - n_a - n_b
    chi_critical = chi2.isf(alpha, df=dof)

    if Q < chi_critical:
        print(f"The residual is white and the estimated order is n_a= {n_a} and n_b = {n_b}")
    else:
        print(f"The residual is not white with n_a={n_a} and n_b={n_b}")

    return Q < chi_critical


def gpac_order_chi_square_test(possible_order_ARMA, train_data, start, stop, lags, actual_outputs):
    results = []
    for n_a, n_b in possible_order_ARMA:
        try:
            # estimate the model parameters
            model = sm.tsa.ARMA(train_data, (n_a, n_b)).fit(trend="nc", disp=0)

            # performing h step predictions
            predictions = model.predict(start=start, end=stop)
            
            # calculate forecast errors
            residuals = np.subtract(actual_outputs, predictions )

            # autocorrelation of residuals
            re = calc_acf(residuals, lags)
            
            # compute Q value for chi square test
            Q = len(actual_outputs) * np.sum(np.square(re[1:]))

            # checking the chi square test
            if chi_square_test(Q, lags, n_a, n_b):
                results.append((n_a, n_b))

        except Exception as e:
            # print(e)
            pass

    return results






