import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from scipy.interpolate import interp1d, splrep, splev
import random
import csv
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


## ================================= Functions ============================================
def interplote_data(list, num):
    x = np.linspace(0,1, len(list))
    spl = splrep(x, list)
    new_x = np.linspace(0,1, num)
    new_value = splev(new_x, spl)
    return new_value

def normalize_data(list):
    list_max = max(list)
    normalized_data_1d = [ (data / list_max)for data in list]
    return normalized_data_1d
    

def interpolate_data_b(x, y, num):
    spl = splrep(x, y)  
    new_x = np.linspace(min(x), max(x), num)  
    new_value = splev(new_x, spl)
    return new_value

def half_data (data_list, num):
    num = int(num)
    front_half = data_list[0:num][::-1]
    back_half = data_list[num:]
    
    return front_half,back_half


def align_timestamp_data(left_max, right_max, data_left_fraction, data_right_fraction, data):

    data_left_index = round(data_left_fraction*len(data))
    data_right_index = round(data_right_fraction*len(data))
    data_left = data[:data_left_index]
    extended_data_left = interplote_data(data_left, left_max)
    extended_data_left = extended_data_left.tolist()

    data_middle = data[data_left_index: (data_right_index+1)]
    extended_data_middle = interplote_data(data_middle, (right_max-left_max))
    extended_data_middle = extended_data_middle.tolist()

    data_right = data[(data_right_index+1):]
    extended_data_right = interplote_data(data_right, (1000 - right_max))
    extended_data_right= extended_data_right.tolist()

    final_data = extended_data_left + extended_data_middle + extended_data_right

    return final_data


def multi_image_alignment(left_max, right_max, data, green_data, cyan_data, magenta_data):

    data_left_index = data.index(max(data[:round(len(data)*0.3)]))
    data_left_fraction = data_left_index/len(data)
    data_right_index = data.index(max(data[round(len(data)*0.7):]))
    data_right_fraction = data_right_index/len(data)


    data_left = data[:data_left_index]
    extended_data_left = interplote_data(data_left, left_max)
    extended_data_left = extended_data_left.tolist()

    data_middle = data[data_left_index: (data_right_index+1)]
    extended_data_middle = interplote_data(data_middle, (right_max-left_max))
    extended_data_middle = extended_data_middle.tolist()

    data_right = data[(data_right_index+1):]
    extended_data_right = interplote_data(data_right, (1000 - right_max))
    extended_data_right= extended_data_right.tolist()

    final_data = extended_data_left + extended_data_middle + extended_data_right
    fina_green_data = align_timestamp_data(left_max, right_max, data_left_fraction, data_right_fraction, green_data)
    fina_cyan_data = align_timestamp_data(left_max, right_max, data_left_fraction, data_right_fraction, cyan_data)
    fina_magenta_data = align_timestamp_data(left_max, right_max, data_left_fraction, data_right_fraction, magenta_data)

    return final_data, fina_green_data, fina_cyan_data, fina_magenta_data


def power_function(x, a, b):
    return a * x**b

def plot_exp (x_data, y_data, i):

    power_params, _ = curve_fit(power_function, x_data, y_data, p0=(1, 0.1))
    x_fit = np.linspace(0, 22, 100) 
    y_power_fit = power_function(x_fit, *power_params)

    plt.scatter(x_data, y_data, color='red', label='Data Points')
    plt.plot(x_fit, y_power_fit, label=f'Power Fit: y={power_params[0]:.3f}x^{power_params[1]:.3f}', color='green')
    plt.xlabel('day')
    plt.ylabel('integral length percentage [0,1]')
    plt.legend()
    plt.title('{}'.format(i+1))
    plt.show()
    return power_params


def reverse_exp_fit_single(y_values, power_params):
    x_values = []
    for y_value in y_values:
        if y_value == 0:
            x_values.append(0)
        else:
            x_value = (y_value/ power_params[0]) ** (1 / power_params[1])
            x_values.append(x_value)
    return x_values



def ave_center_data(data):
    n = len(data)
    middle_1_percent = data[700:750]

    baseline = sum(middle_1_percent) / len(middle_1_percent) 
    
    return baseline


def fig_6_create_single (new_red, num_split, power_params_left, power_params_right):
    
   
    # num_split = len(new_red)/2
    front_red,back_red = half_data (new_red, num_split)

    intensity_front = interplote_data (front_red, 1000)
    intensity_back = interplote_data (back_red, 1000)

    y_values = np.linspace(0, 1, 1000).tolist()
    x_values = reverse_exp_fit_single(y_values, power_params_left)
    intensity_front = interpolate_data_b (x_values, intensity_front, 1000)

    y_values = np.linspace(0, 1, 1000).tolist()
    x_values = reverse_exp_fit_single(y_values, power_params_right)
    intensity_back = interpolate_data_b (x_values, intensity_back, 1000)

    baseline_front = np.mean(intensity_front[750:780]) 
    intensity_front = [(x/baseline_front) - 1 for x in intensity_front]

    baseline_back = np.mean(intensity_back[750:780]) 
    intensity_back = [(x/baseline_back) - 1 for x in intensity_back]
    
    return intensity_front, intensity_back


## ==================================== Scripts =========================================

arc_path = 'arc1.csv'

arc1_path = 'arc1-1.csv'

cyan_path = 'cyan1.csv'

egr_path = 'Egr1.csv'

green_path = 'green1.csv'

magenta_path = 'magenta1.csv'


## Align the same image signals took in different rounds of immunostaining

df = pd.read_csv(arc_path)
new_arc = df.iloc[:, 1].tolist() 

df = pd.read_csv(arc1_path)
new_arc11 = df.iloc[:, 1].tolist() 
new_arc11 = interplote_data(new_arc11, 1000)
new_arc11 = normalize_data(new_arc11)

df = pd.read_csv(cyan_path)
new_cyan = df.iloc[:, 1].tolist() 

df = pd.read_csv(egr_path)
new_egr = df.iloc[:, 1].tolist() 
new_egr = interplote_data(new_egr, 1000)
new_egr = normalize_data(new_egr)

df = pd.read_csv(green_path)
new_green = df.iloc[:, 1].tolist() 

df = pd.read_csv(magenta_path)
new_magenta = df.iloc[:, 1].tolist() 

# alignment function
left_max = new_arc11.index(max(new_arc11[:300]))
right_max = new_arc11.index(max(new_arc11[700:]))
final_arc_data, final_green_data, final_cyan_data, final_magenta_data = multi_image_alignment(left_max, right_max, new_arc, new_green, new_cyan, new_magenta)

## show alignment result and identifying timestamp points based on it 
final_arc_data = normalize_data(final_arc_data)
final_green_data = normalize_data(final_green_data)
final_cyan_data = normalize_data(final_cyan_data)
final_magenta_data = normalize_data(final_magenta_data)

plt.plot(final_arc_data, color = 'red', alpha = 0.5)
plt.plot(new_arc11, color = 'red', alpha = 1)
plt.plot(final_cyan_data, color = 'cyan', alpha = 1)
plt.plot(new_egr, color = 'blue', alpha = 1)
plt.plot(final_green_data, color = 'green', alpha = 1)
plt.plot(final_magenta_data, color = 'magenta', alpha = 1)

# here's the example of timepoints for timestamps 
plt.axvline(x=135, linestyle='--', color='green', alpha=0.5, label="left timepoint 2")
plt.axvline(x=210, linestyle='--', color='green', alpha=0.5, label="left timepoint 1")
plt.axvline(x=870, linestyle='--', color='blue', alpha=0.5, label="right timepoint 1")
plt.axvline(x=916, linestyle='--', color='blue', alpha=0.5, label="right timepoint 2")
plt.axvline(x=10, linestyle='--', color='gray', alpha=0.5, label="left end")
plt.axvline(x=982, linestyle='--', color='gray', alpha=0.5, label="right end")

plt.title(f'find timestamp point based on aligned data')
plt.legend()
plt.show()


## here we stored all the manual-observed switching point in a file 'timestamp point along fiber.csv' 

## find optimal split point
time_path = 'timestamp point along fiber.csv'

split_pointssss = range (300, 701)
correlationssss = []
# Arc data
df = pd.read_csv(arc1_path)
new_arc11 = df.iloc[:, 1].tolist() 
new_arc = interplote_data(new_arc11, 1000)

# Egr data
df = pd.read_csv(egr_path)
new_egr = df.iloc[:, 1].tolist() 
new_egr = interplote_data(new_egr, 1000)

# Timestamp data
df_time = pd.read_csv(time_path)
row_values = df_time.iloc[0]
values_list = row_values.tolist()

for split_point in split_pointssss:

    try:
        num1 = int(values_list[4])
        num2 = int(values_list[5])

        left1 = (split_point - values_list[0])/(split_point-num1) # bigger
        left2 = (split_point - values_list[1])/(split_point - num1)
        right1 = (values_list[2] - split_point)/(num2 - split_point)
        right2 = (values_list[3] - split_point)/(num2 - split_point) # bigger

        new_arc = interplote_data(new_arc[num1:(num2+1)], 1000)
        new_egr = interplote_data(new_egr[num1:(num2+1)], 1000)

        x_data = np.array([18, 20, 21])
        y_data = np.array([left2, left1, 1])  
        power_params_left, _ = curve_fit(power_function, x_data, y_data, p0=(1, 0.1))

        y_data = np.array([right1, right2, 1])  
        power_params_right, _ = curve_fit(power_function, x_data, y_data, p0=(1, 0.1))


        intensity_front_arc, intensity_back_arc = fig_6_create_single (new_arc, split_point, power_params_left, power_params_right)
        intensity_front_egr, intensity_back_egr = fig_6_create_single (new_egr, split_point, power_params_left, power_params_right)

        r_arc, p_value = pearsonr(intensity_front_arc[700:], intensity_back_arc[700:])
        r_egr, p_value = pearsonr(intensity_front_egr[700:], intensity_back_egr[700:])


        correlation = r_arc + r_egr
        correlationssss.append(correlation)
    except RuntimeError as e:
        correlationssss.append(-100)
        print('failed')

optimal_index = correlationssss.index(max(correlationssss))
split_point = split_pointssss[optimal_index]
print(f'optimal split point is {split_point}')


## recover the fluorescent intensity profiles with the real time

# Arc data
df = pd.read_csv(arc1_path)
new_arc11 = df.iloc[:, 1].tolist() 
new_arc = interplote_data(new_arc11, 1000)

# Egr data
df = pd.read_csv(egr_path)
new_egr = df.iloc[:, 1].tolist() 
new_egr = interplote_data(new_egr, 1000)

# Timestamp data
df_time = pd.read_csv(time_path)
row_values = df_time.iloc[0]
values_list = row_values.tolist()

num1 = int(values_list[4])
num2 = int(values_list[5])
left1 = (split_point - values_list[0])/(split_point-num1) # bigger
left2 = (split_point - values_list[1])/(split_point - num1)
right1 = (values_list[2] - split_point)/(num2 - split_point)
right2 = (values_list[3] - split_point)/(num2 - split_point) # bigger

new_arc = interplote_data(new_arc[num1:(num2+1)], 1000)
new_egr = interplote_data(new_egr[num1:(num2+1)], 1000)

x_data = np.array([18, 20, 21])
y_data = np.array([left2, left1, 1])  
power_params_left, _ = curve_fit(power_function, x_data, y_data, p0=(1, 0.1))

y_data = np.array([right1, right2, 1])  
power_params_right, _ = curve_fit(power_function, x_data, y_data, p0=(1, 0.1))


intensity_front_arc, intensity_back_arc = fig_6_create_single (new_arc, split_point, power_params_left, power_params_right)
intensity_front_egr, intensity_back_egr = fig_6_create_single (new_egr, split_point, power_params_left, power_params_right)

arc_intensity = [(intensity_front_arc[k] + intensity_back_arc[k])/2 for k in range(len(intensity_front_arc))]
egr_intensity = [(intensity_front_egr[k] + intensity_back_egr[k])/2 for k in range(len(intensity_front_egr))]

x_values =  np.linspace(0, 21, 1000).tolist()
plt.plot(x_values, arc_intensity, color = '#cb1010', label = 'arc')
plt.plot(x_values, egr_intensity, color = '#313131', label = 'egr1')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim(16, 21)
plt.yticks([0, 1, 2])
plt.xticks([17, 18, 19, 20, 21])
plt.xlabel("Day")
plt.ylabel("Signal relative change from baseline")
plt.legend()
plt.show() 

