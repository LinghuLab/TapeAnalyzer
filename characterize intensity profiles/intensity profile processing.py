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
import emoji
import csv


## ===================================== Functions =========================================

def adjust_the_line(red_cell, green_cell):
    
    # for the red values
    red_values = [value.strip() for value in red_cell.split(',')]
    # Append each value as a new row
    red_value = [float(num) for num in red_values]
    new_red_value = interplote_data(red_value, 10000)

    average = lowess_smooth(new_red_value)

    baseline_average = find_baseline(average)
    mini_crop, max_crop = find_crops(average, baseline_average)
    data_new = average[mini_crop: max_crop]
    # Perform interpolation
    x = np.linspace(0,1, len(data_new))
    y = data_new
    spl = splrep(x, y)  # Fit the spline
    new_x = np.linspace(x.min(), x.max(), 10000)  
    new_red_y = splev(new_x, spl) 

    # For the green values
    green_values = [value.strip() for value in green_cell.split(',')]
    # Append each value as a new row
    green_value = [float(num) for num in green_values]
    new_green_value = interplote_data(green_value, 10000)
    
    smooth_green = lowess_smooth(new_green_value)
    data_new_green = smooth_green[mini_crop: max_crop]
    # Perform interpolation
    x = np.linspace(0,1, len(data_new_green))
    y = data_new_green
    spl = splrep(x, y)  # Fit the spline
    new_x = np.linspace(x.min(), x.max(), 10000)  
    new_green_y = splev(new_x, spl) 

    
    return new_red_y,new_green_y


def interplote_data(list, num):
    x = np.linspace(0,1, len(list))
    spl = splrep(x, list)
    new_x = np.linspace(0,1, num)
    new_value = splev(new_x, spl)
    return new_value


        
def find_baseline(data):
    n = len(data)

    start_index = int(n * 0.30)
    end_index = int(n * 0.70)
    middle_40_percent = data[start_index:end_index]
    min_value = min(middle_40_percent)
    min_indices_relative = [i for i, value in enumerate(middle_40_percent) if value == min_value]
    min_indices_absolute = start_index + min_indices_relative[0]

    baseline_value = data[min_indices_absolute-int(n*0.005):min_indices_absolute+int(n*0.005)]
    baseline_average = np.mean(baseline_value)
    return baseline_average


def find_crops(data, target_value):
    data_as_int = [int(float(x)) for x in data]
    matching_indices = [i for i, value in enumerate(data_as_int) if value<target_value]
    mini_crop = 0
    max_crop = len(data)
    if 0 in matching_indices:
        i = 0
        while i in range(0,len(matching_indices)):
            if 0<(matching_indices[i+1]-matching_indices[i]) <=2:
                mini_crop = matching_indices[i+1] 
                i += 1
            else:
                break

    reverse_matching_indices = matching_indices[::-1]

    if (len(data)-1) in reverse_matching_indices:
        i = 0
        while i in range(0,len(reverse_matching_indices)):
            if 0<(reverse_matching_indices[i]-reverse_matching_indices[i+1]) <=2:
                max_crop = reverse_matching_indices[i+1] 
                i += 1
            else:
                break

    return mini_crop, max_crop

    

def lowess_smooth (data):
    x = np.linspace(0, 1, len(data))
    y = data

    frac = 0.1  
    lowess_result = lowess(y, x, frac=frac)

    return lowess_result[:,1]

def normalize_data(list):
    list_max = max(list)
    normalized_data_1d = [ (data / list_max)for data in list]
    return normalized_data_1d

def interpolate_data_b(x, y, num):
    spl = splrep(x, y)  
    new_x = np.linspace(min(x), max(x), num)  
    new_value = splev(new_x, spl)
    return new_value


def fraction (list):
    total_sum = sum(list)
    cumulative_sum = []
    current_sum = 0
    for value in list:
        current_sum += value
        cumulative_sum.append(current_sum)
    cumulative_weights = [(cs / total_sum)for cs in cumulative_sum]
    return cumulative_weights


def calculate_plot_parameter(intensities):
    average = np.mean(intensities, axis=0)
    average = lowess_smooth(average)

    stderr = np.std(intensities, axis=0) / np.sqrt(len(intensities))
    stderr = lowess_smooth(stderr)

    std = np.std(intensities, axis=0)
    std = lowess_smooth(std)

    return average, stderr, std


def half_data (list, split_point):
    front_half = list[:split_point]
    back_half = list[split_point:]
    front_half_reversed = front_half[::-1]
    
    return front_half_reversed,back_half

def ave_center_data(data, split_point):
    n = len(data)

    start = round(split_point - n * 0.005)
    end = round(split_point + n * 0.005)
    middle_1_percent = data[start:end]

    baseline = sum(middle_1_percent) / len(middle_1_percent) 
    
    return baseline

def fig_4_create_split (new_red,new_green, split_point):
    
    # Green
    front_green,back_green = half_data (new_green, split_point)

    x_front = fraction (front_green)

    x_back = fraction (back_green)

    # Red

    baseline = ave_center_data (new_red, split_point)
    front_red,back_red = half_data (new_red, split_point)

    front_red_b = [(x/baseline) - 1 for x in front_red]
    back_red_b = [(x/baseline) - 1 for x in back_red]

    intensity_front = interpolate_data_b(x_front, front_red_b, 10000)
    intensity_back = interpolate_data_b(x_back, back_red_b, 10000)

    return intensity_front, intensity_back

def figure_3_create_split(new_red,new_green, split_point):
    # Green
    front_green,back_green = half_data (new_green, split_point)
    x_front = fraction (front_green)
    x_back = fraction (back_green)

    # Red
    front_red,back_red = half_data (new_red, split_point)

    #Integrate
    intensity_front = interpolate_data_b (x_front, front_red, 10000)
    intensity_front = normalize_data(intensity_front)

    intensity_back = interpolate_data_b (x_back, back_red, 10000)
    intensity_back = normalize_data(intensity_back)

    return intensity_front, intensity_back


from scipy.stats import pearsonr

def find_optimal_point(new_red, new_green):
    correlations = []

    split_lists = range(3000, 7000, 10)

    for split_point in split_lists:
        try:
            intensity_front, intensity_back = fig_4_create_split (new_red,new_green, split_point)
            r, _ = pearsonr(intensity_front, intensity_back)
            correlations.append(r)
        except Exception as e:
            correlations.append(0)
        
    optimal_index = correlations.index(max(correlations))
    optimal_split_points = split_lists[optimal_index]

    return optimal_split_points


    

def allinfunction_optimal_split(file_path):
    df = pd.read_csv(file_path)

    red_values_list = []
    green_values_list = []
    fig_3_intensities = []
    fig_4_intensities = []
    print('analyzing...')

    for i in range(0,len(df['Red_intensity'])): 
        try:
            red_cell = df['Red_intensity'][i]
            green_cell = df['Green_intensity'][i]
            new_red,new_green = adjust_the_line(red_cell, green_cell)

            split_point = find_optimal_point(new_red, new_green)

            # This is figure 1 & 2 parts
            normalized_red_value = normalize_data(new_red)
            normalized_green_value = normalize_data(new_green)
            red_values_list.append(normalized_red_value)
            green_values_list.append(normalized_green_value)

            # This is figure 3 part
            intensity_front_3, intensity_back_3 = figure_3_create_split(new_red,new_green, split_point)
            intensity_3 = [(intensity_front_3[b] + intensity_back_3[b])/2 for b in range(len(intensity_front_3))]
            fig_3_intensities.append(intensity_3)
        
            # This is figure 4 part
            intensity_front_4, intensity_back_4 = fig_4_create_split (new_red,new_green, split_point)
            intensity_4 = [(intensity_front_4[b] + intensity_back_4[b])/2 for b in range(len(intensity_front_4))]
            fig_4_intensities.append(intensity_4)

        except Exception as e:
            print('error in {}'.format(i+1))


    return red_values_list, green_values_list, fig_3_intensities, fig_4_intensities






def plot_1(red_values_list, average, stderr, std):
    red = '#cc493e'
    for data in red_values_list:
        plt.plot(data, color='#b6b4b4', alpha=0.65,linewidth=0.5)
        
    plt.plot(average, color=red, linewidth=3, label='Average')
    plt.fill_between(range(len(average)), average - stderr, average + stderr, color=red, alpha=0.25)
    plt.fill_between(range(len(average)), average - std, average + std, color=red, alpha=0.115)
    plt.xticks([0, len(average)//2, len(average)-1], [-0.5, 0, 0.5])
    plt.yticks([0, 1])

    plt.xlim(0, len(average)-1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


    plt.xlabel("Location along XRI")
    plt.ylabel("Red intensity")

    plt.savefig('output.png')

    plt.show()
    

def plot_2 (green_values_list, average, stderr, std):
    green = '#4a8b3c' 
    for data in green_values_list:
        plt.plot(data, color='#b6b4b4', alpha=0.65,linewidth=0.5)
        
    plt.plot(average, color= green, linewidth=3, label='Average')
    plt.fill_between(range(len(average)), average - stderr, average + stderr, color= '#a2c873', alpha=0.45)
    plt.fill_between(range(len(average)), average - std, average + std, color= '#a2c873', alpha=0.25)
    plt.xticks([0, len(average)//2, len(average)-1], [-0.5, 0, 0.5])
    plt.yticks([0, 1])

    plt.xlim(0, len(average)-1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel("Location along XRI")
    plt.ylabel("Red intensity")

    plt.savefig('output.png')


    plt.show()


def plot_3 (intensities, average, stderr, std):
    for data in intensities:
        plt.plot(data, color='#b6b4b4', alpha=0.65,linewidth=0.5)
        
    plt.plot(average, color='#313131', linewidth=3, label='Average')
    plt.fill_between(range(len(average)), average - stderr, average + stderr, color='gray', alpha=0.45)
    plt.fill_between(range(len(average)), average - std, average + std, color='gray', alpha=0.15)
    plt.xticks([0, len(average)//2, len(average)-1], [0, 0.5, 1])

    plt.yticks([0, 1])

    plt.xlim(0, len(average)-1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel("Fraction of green intensity line integral from center")
    plt.ylabel("Recovered red intensity")

    plt.show()


def plot_4 (intensities, average, stderr, std):  
    for data in intensities:
        plt.plot(data, color='#b6b4b4', alpha=0.65,linewidth=0.5)

    plt.plot(average, color='#313131', linewidth=3, label='Average')
    plt.fill_between(range(len(average)), average - stderr, average + stderr, color='gray', alpha=0.45)
    plt.fill_between(range(len(average)), average - std, average + std, color='gray', alpha=0.15)

    plt.xticks([0, len(average)//2, len(average)-1], [0, 0.5, 1])
    plt.yticks([0, 1])
    plt.xlim(0, len(average)-1)
    plt.ylim(-0.25, 1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel("Fraction of green intensity line integral from center")
    plt.ylabel("Red intensity change from baseline")


    plt.show()


## ===================================== Scripts =========================================

file_path = 'D:\\summer_research2023\\Lirong paper\\Lirong paper code final\\Characterize intensity profiles\\SARE_ArcMin KCl 3h.csv'

red_values_list, green_values_list, fig_3_intensities, fig_4_intensities_3h = allinfunction_optimal_split(file_path)

# Plot 1
average, stderr, std = calculate_plot_parameter(red_values_list)
plot_1 (red_values_list, average, stderr, std)

# Plot 2
average, stderr, std = calculate_plot_parameter(green_values_list)
plot_2 (green_values_list, average, stderr, std)

# Plot 3
average, stderr, std = calculate_plot_parameter(fig_3_intensities)
plot_3 (fig_3_intensities, average, stderr, std)

# Plot 4
average_3h, stderr_3h, std_3h = calculate_plot_parameter(fig_4_intensities_3h)
plot_4 (fig_4_intensities_3h, average_3h, stderr_3h, std_3h)
