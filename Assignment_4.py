# %% [markdown]
# # Assignment 4 - Proposal for a new client
# 
# Your firm is trying to engage a new client. They want to improve their catchment model to better understand the water quality in Homebush Bay. Previously, a stormwater quality model has been developed for the 8 km2 Powells Creek catchment, which drains into Homebush Bay. The basis of this original stormwater quality model is the US EPA SWMM software.
# 
# The SWMM model uses a non-linear reservoir model for surface runoff (pond model) with a kinematic model for flows in the channel. Water quality simulation is based on simulation of the EMC (event mean concentration) rather than a pollutograph or loadograph. The focus of the water quality simulation is on the prediction of the total mass of contaminant transported during the storm event.  At present only the flow quantity component of the model is being calibrated. The client would like you to demonstrate the ability to create a water quality model and provide an indicated of the future work that might be required.
# 
# The water quality component of the model is yet to be developed.
# 
# The client has been very generous and shared the previous study.
# 
# Data available for use in the calibration are:
# 
# 1. Rainfall data
#     1. 6 pluviometers (i.e. continuous rain gauges) within the catchment;
#     2. 6 pluviometers adjacent to the catchment;
#     3. 1 daily read gauge within the catchment and another 6 in close proximity to the catchment; and
#     4. Radar images of rainfall over the catchment; these images have not been processed, (i.e. converted to rainfall depths) but they are available for the study. The resolution of these radar images is 1 km x 1 km.
# 2. Catchment data
#     1. SRTM and LIDAR based models of the topography;
#     2. Detailed plans of the stormwater system in the catchment (note that the catchment is serviced by separate stormwater and sanitary sewer systems);
#     3. Detailed maps in GIS format of land use in the catchment (note that the whole catchment is urbanised); Soil Maps of the catchment derived from the National Soil Survey
# 
# Gauged data at one site in the catchment for 49 years. The client indicated that the approving athorities identied concerns about the rating curve at this gauge station for higher flows. This concern arises from the proximity of railway culverts and the potential for the station to be drowned during higher flows.  This gauging station is located in the upper reaches of the catchment and monitors only 2.4 km2 of the catchment.
# 
# You are required to provide a detailed proposal to the client (maximum of 20 pages, includive of diagrams, drawings, references and appendices).
# 
# The file below contains the details of calibration simulations.  Data included in the file for each event (there are multiple events in the file) are:
# 
# * Rainfall records from the pluviometer located at the catchment outlet;
# * Recorded flow for the event; and
# * Predicted flows for alternative sets of parameter values.
# 
# To prepare you have had a detailed discussion with your manager. The questions that they would like you to addresss are as follows:
# 
# *Firstly*
# 
# 1. Select and justify a calibration metric suitable for use to assess the calibration of the quantity component of the model;
# 2. Apply this metric to the available data in a manner where all events are considered concurrently. You will need to consider how the different events can be combined and to justify this approach;
# 3. Select and justify the best parameter set to be used with the water quality component of the model.
# 
# *Secondly*
# 
# 1. How the different conceptual components of a catchment model would be used in developing this model.
# 2. The sources of error in predictions obtained from a catchment model with reference to the current study. How should knowledge of these errors be considered in the current application?
# 3. Whether the selected parameter values would change if a kinematic wave model were to be used in lieu of the non-linear reservoir model of surface runoff.
# 
# *Finally*
# 
# You have been asked to improve the current catchment model. To achieve that improvement, you have time and resources available.  What would you do?
# 
# Available data
# 
# `Recorded and Predicted Flows`

# %%
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import datetime as dt
import math
import os

# %% [markdown]
# # inputs

# %%
arg_input_data = './Inputs_CM_Assignment_4/49255 2021 Assignment 4 data.xlsx'
arg_output_dir = './Outputs_CM_Assignment_4/'

print(
    arg_input_data,
    arg_output_dir,
    sep='\n')

# %%
sheet = ['Event {}'.format(ind) for ind in range(1, 4)]
cols = ['Date_Time_recorded', 'Q_recorded', 'Date_Time_predicted', 'Q_predicted_1', 'Q_predicted_2', 'Q_predicted_3', 'Q_predicted_4']

print(
    sheet,
    cols,
    sep='\n')

# %% [markdown]
# # output directory

# %%
def create_output_dir(arg_output_dir):
    """create output directory if it does not exist

    arguments:
        arg_output_dir = [string] './Outputs_CM_Assignment_2/'
    """
    if not os.path.exists(arg_output_dir):
        os.makedirs(arg_output_dir)

# %%
create_output_dir(arg_output_dir)

# %% [markdown]
# # Reading XLSX files

# %%
data = pd.DataFrame(data=sheet, columns=['sheet'])
data['data'] = data.sheet.apply(func=lambda arg: pd.read_excel(io=arg_input_data, sheet_name=arg, names=cols, skiprows=1))
data.sheet = data.sheet.str.split(pat=' ').str.join(sep='_').str.lower()
data

# %%
data.data[0]

# %%
data.data[1]

# %%
data.data[2]

# %% [markdown]
# # Slicing XLSX files

# %%
def recorded_data(arg_df):
    arg_df = arg_df[['Date_Time_recorded', 'Q_recorded']].copy()
    arg_df.dropna(axis=0, inplace=True)
    arg_df.set_index(keys='Date_Time_recorded', inplace=True)
    arg_df.index.name = 'Date_Time'
    return arg_df

# %%
data['recorded_data'] = data.data.apply(func=lambda arg: recorded_data(arg))
data

# %%
data.recorded_data[0]

# %%
def predicted_data(arg_df):
    arg_df = arg_df[['Date_Time_predicted', 'Q_predicted_1', 'Q_predicted_2', 'Q_predicted_3', 'Q_predicted_4']].copy()
    arg_df.dropna(axis=0, inplace=True)
    arg_df.set_index(keys='Date_Time_predicted', inplace=True)
    arg_df.index.name = 'Date_Time'
    return arg_df

# %%
data['predicted_data'] = data.data.apply(func=lambda arg: predicted_data(arg))
data

# %%
data.predicted_data[0]

# %%
def flows(arg_recorded, arg_predicted):
    arg_df = pd.merge(left=arg_recorded, right=arg_predicted, how='inner', left_index=True, right_index=True)
    return arg_df

# %%
data['flow_rate'] = data.apply(func=lambda arg: flows(arg.recorded_data, arg.predicted_data), axis=1)
data

# %%
data.flow_rate[0]

# %%
data['starting_time'] = data.flow_rate.apply(func=lambda arg: arg.index[0])
data['finishing_time'] = data.flow_rate.apply(func=lambda arg: arg.index[-1])
data['storm_duration'] = data.finishing_time - data.starting_time
data

# %%
def export_to_csv(arg_flow_rate, arg_output_dir, arg_sheet):
    arg_flow_rate.to_csv(path_or_buf='./{}data_{}.csv'.format(arg_output_dir, arg_sheet))

# %%
data.apply(func=lambda arg: export_to_csv(arg.flow_rate, arg_output_dir, arg.sheet), axis=1)

# %% [markdown]
# # Plotting

# %%
def plotting(arg_flow_rate, arg_sheet, arg_output_dir, arg_starting_time, arg_finishing_time, arg_storm_duration):
    fig, ax = plt.subplots(figsize=(15,8))
    ax = arg_flow_rate.plot(
        kind='line',
        ax=ax,
        title='{}\nstarting time: {}\nfinishing time: {}\nstorm duration: {}\nRecorded and Predicted flows\nFlow Rate vs. Date Time'.format(
            ' '.join(arg_sheet.capitalize().split(sep='_')),
            arg_starting_time,
            arg_finishing_time,
            arg_storm_duration
            ),
        xlabel='date time',
        ylabel='flow rate ($m^3/s$)'
        )
    ax.grid(visible=True, which='both')
    fig.savefig('./{}data_{}.png'.format(arg_output_dir, arg_sheet))

# %%
data.apply(func=lambda arg: plotting(arg.flow_rate, arg.sheet, arg_output_dir, arg.starting_time, arg.finishing_time, arg.storm_duration), axis=1)

# %% [markdown]
# # Objective functions

# %%
def relative_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    max_rec_flow_rate = arg_recorded.max()
    max_pre_flow_rate = arg_predicted.max()
    
    # relative error (RE)
    re = abs(max_pre_flow_rate - max_rec_flow_rate)/max_rec_flow_rate

    arg_df = pd.concat(objs=[arg_df[arg_recorded == max_rec_flow_rate], arg_df[arg_predicted == max_pre_flow_rate]])
    arg_df.drop_duplicates(inplace=True)
    arg_df.sort_index(inplace=True)
    arg_df.to_csv(path_or_buf='{}df__{}__RE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return re, arg_df

# %%
def absolute_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    max_rec_flow_rate = arg_recorded.max()
    max_pre_flow_rate = arg_predicted.max()
    
    # absolute error (AE)
    ae = abs(max_pre_flow_rate - max_rec_flow_rate)

    arg_df = pd.concat(objs=[arg_df[arg_recorded == max_rec_flow_rate], arg_df[arg_predicted == max_pre_flow_rate]])
    arg_df.drop_duplicates(inplace=True)
    arg_df.sort_index(inplace=True)
    arg_df.to_csv(path_or_buf='{}df__{}__AE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return ae, arg_df

# %%
def time_to_peak_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    peak_rec_date_time = arg_recorded[arg_recorded == arg_recorded.max()].index
    peak_pre_date_time = arg_predicted[arg_predicted == arg_predicted.max()].index
    
    # time to peak (TtoP)
    ttop = abs(peak_pre_date_time - peak_rec_date_time)
    ttop = pd.DataFrame(index=ttop)
    ttop.reset_index(inplace=True)
    ttop = ttop[ttop.columns[0]][0]

    arg_df = pd.concat(objs=[arg_df.loc[peak_rec_date_time], arg_df.loc[peak_pre_date_time]])
    arg_df.drop_duplicates(inplace=True)
    arg_df.sort_index(inplace=True)
    arg_df.to_csv(path_or_buf='{}df__{}__TtoP__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return ttop, arg_df

# %%
def time_to_centroid_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):

    arg_rec = pd.concat(objs=[arg_recorded.iloc[[0]], arg_recorded, arg_recorded.iloc[[-1]]])
    arg_rec.iloc[0] = 0
    arg_rec.iloc[-1] = 0
    arg_rec = pd.DataFrame(data=arg_rec)
    arg_rec.reset_index(inplace=True)
    arg_rec_0 = arg_rec[arg_rec.columns[0]].iloc[0]
    arg_rec['duration_min'] = (arg_rec[arg_rec.columns[0]] - arg_rec_0)/pd.Timedelta(minutes=1)
    arg_rec['coord'] = arg_rec.apply(func=lambda arg: (arg['duration_min'], arg[arg_rec.columns[1]]), axis=1)
    centroid_rec_date_time = Polygon(arg_rec.coord.to_list()).centroid.x
    centroid_rec_date_time = arg_rec_0 + pd.Timedelta(minutes=centroid_rec_date_time)

    arg_pre = pd.concat(objs=[arg_predicted.iloc[[0]], arg_predicted, arg_predicted.iloc[[-1]]])
    arg_pre.iloc[0] = 0
    arg_pre.iloc[-1] = 0
    arg_pre = pd.DataFrame(data=arg_pre)
    arg_pre.reset_index(inplace=True)
    arg_pre_0 = arg_pre[arg_pre.columns[0]].iloc[0]
    arg_pre['duration_min'] = (arg_pre[arg_pre.columns[0]] - arg_pre_0)/pd.Timedelta(minutes=1)
    arg_pre['coord'] = arg_pre.apply(func=lambda arg: (arg['duration_min'], arg[arg_pre.columns[1]]), axis=1)
    centroid_pre_date_time = Polygon(arg_pre.coord.to_list()).centroid.x
    centroid_pre_date_time = arg_pre_0 + pd.Timedelta(minutes=centroid_pre_date_time)

    # time to centroid (TtoC)
    ttoc = abs(centroid_pre_date_time - centroid_rec_date_time)

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)
    
    arg_df_rec = arg_df.iloc[
        np.where(arg_recorded.index > centroid_rec_date_time)[0][0] - 1:
        np.where(arg_recorded.index > centroid_rec_date_time)[0][0] + 1
        ].copy()
    
    arg_df_rec.loc[centroid_rec_date_time] = [np.nan, np.nan]
    arg_df_rec.drop_duplicates(inplace=True)
    arg_df_rec.sort_index(inplace=True)
    arg_df_rec.interpolate(method='time', axis=0, inplace=True)
    
    arg_df_pre = arg_df.iloc[
        np.where(arg_predicted.index > centroid_pre_date_time)[0][0] - 1:
        np.where(arg_predicted.index > centroid_pre_date_time)[0][0] + 1
        ].copy()
    
    arg_df_pre.loc[centroid_pre_date_time] = [np.nan, np.nan]
    arg_df_pre.drop_duplicates(inplace=True)
    arg_df_pre.sort_index(inplace=True)
    arg_df_pre.interpolate(method='time', axis=0, inplace=True)

    arg_df = pd.concat(objs=[arg_df_pre, arg_df_rec], axis=0)
    arg_df.drop_duplicates(inplace=True)
    arg_df.sort_index(inplace=True)

    arg_df.to_csv(path_or_buf='{}df__{}__TtoC__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return ttoc, arg_df

# %%
def sum_of_absolute_difference_abs_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: mean absolute deviation - absolute terms

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_abs_diff'] = abs(arg_df[arg_predicted.name] - arg_df[arg_recorded.name])

    # sum of absolute difference - absolute terms (SofAD_abs)
    sofad_abs = arg_df.Q_abs_diff.sum()

    arg_df.to_csv(path_or_buf='{}df__{}__SofAD_abs__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return sofad_abs, arg_df

# %%
def sum_of_absolute_difference_rel_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: mean absolute deviation - relative terms

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_abs_diff'] = abs(arg_df[arg_predicted.name] - arg_df[arg_recorded.name])
    arg_df['Q_rel_diff'] = arg_df.Q_abs_diff/arg_df[arg_recorded.name]

    # sum of absolute difference - absolute terms (SofAD_rel)
    sofad_rel = arg_df.Q_rel_diff.sum()

    arg_df.to_csv(path_or_buf='{}df__{}__SofAD_rel__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return sofad_rel, arg_df

# %%
def sum_of_the_squared_error_abs_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: sum of squares of the differences - absolute terms

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_sqr_diff'] = arg_df.Q_diff**2

    # sum of the squared error - absolute terms (SoftheSE_abs)
    softhese_abs = arg_df.Q_sqr_diff.sum()

    arg_df.to_csv(path_or_buf='{}df__{}__SoftheSE_abs__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return softhese_abs, arg_df

# %%
def sum_of_the_squared_error_rel_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: sum of squares of the differences - relative terms

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_rel_diff'] = arg_df.Q_diff/arg_df[arg_recorded.name]
    arg_df['Q_sqr_rel_diff'] = arg_df.Q_rel_diff**2

    # sum of the squared error - relative terms (SoftheSE_rel)
    softhese_rel = arg_df.Q_sqr_rel_diff.sum()

    arg_df.to_csv(path_or_buf='{}df__{}__SoftheSE_rel__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return softhese_rel, arg_df

# %%
def mean_squared_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_sqr_diff'] = arg_df.Q_diff**2

    # mean squared error (MSE)
    mse = arg_df.Q_sqr_diff.mean()

    arg_df.to_csv(path_or_buf='{}df__{}__MSE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return mse, arg_df

# %%
def root_mean_squared_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_sqr_diff'] = arg_df.Q_diff**2

    # root mean squared error (RMSE)
    rmse = arg_df.Q_sqr_diff.mean()**0.5

    arg_df.to_csv(path_or_buf='{}df__{}__RMSE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return rmse, arg_df

# %%
def mean_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: bias

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]

    # mean error (ME)
    me = arg_df.Q_diff.mean()

    arg_df.to_csv(path_or_buf='{}df__{}__ME__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return me, arg_df

# %%
def mean_absolute_error_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: bias

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_abs_diff'] = arg_df.Q_diff.abs()

    # mean absolute error (MAE)
    mae = arg_df.Q_abs_diff.mean()

    arg_df.to_csv(path_or_buf='{}df__{}__MAE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return mae, arg_df

# %%
def percent_bias_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: bias

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]

    # percent bias (PBIAS)
    pbias = -arg_df.Q_diff.sum()*100/arg_df[arg_recorded.name].sum()

    arg_df.to_csv(path_or_buf='{}df__{}__PBIAS__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return pbias, arg_df

# %%
def error_variance_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_sqr_diff'] = arg_df.Q_diff**2

    # variance (S2)
    s2 = arg_df.Q_sqr_diff.sum()/(len(arg_df)-1)

    arg_df.to_csv(path_or_buf='{}df__{}__S2__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return s2, arg_df

# %%
def nash_sutcliffe_efficiency_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: modelling efficiency

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_diff'] = arg_df[arg_predicted.name] - arg_df[arg_recorded.name]
    arg_df['Q_sqr_diff'] = arg_df.Q_diff**2
    arg_df['Q_var_rec'] = arg_df[arg_recorded.name] - arg_df[arg_recorded.name].mean()
    arg_df['Q_sqr_var_rec'] = arg_df.Q_var_rec**2

    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - arg_df.Q_sqr_diff.sum()/arg_df.Q_sqr_var_rec.sum()

    arg_df.to_csv(path_or_buf='{}df__{}__NSE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return nse, arg_df

# %%
def kling_gupta_efficiency_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: Kling-Gupta efficacy index - modified

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_var_rec'] = arg_df[arg_recorded.name] - arg_df[arg_recorded.name].mean()
    arg_df['Q_var_pre'] = arg_df[arg_predicted.name] - arg_df[arg_predicted.name].mean()
    arg_df['Q_sqr_var_rec'] = arg_df.Q_var_rec**2
    arg_df['Q_sqr_var_pre'] = arg_df.Q_var_pre**2
    arg_df['Q_prod_var_rec_pre'] = arg_df.Q_var_rec*arg_df.Q_var_pre
    
    # coefficient of determination (R2)
    r2 = (arg_df.Q_prod_var_rec_pre.sum()**2)/(arg_df.Q_sqr_var_rec.sum()*arg_df.Q_sqr_var_pre.sum())
    # correlation coefficient (R)
    r = r2**0.5
    # bias ratio = mean_predicted/mean_recorded
    beta = arg_df[arg_predicted.name].mean()/arg_df[arg_recorded.name].mean()
    # coefficient of variation = std/mean
    alpha = (
        (arg_df[arg_predicted.name].std()/arg_df[arg_predicted.name].mean())/
        (arg_df[arg_recorded.name].std()/arg_df[arg_recorded.name].mean())
    )
    # Kling-Gupta Efficiency (KGE)
    kge = 1 - ((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)**0.5

    arg_df.to_csv(path_or_buf='{}df__{}__KGE__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return kge, arg_df

# %%
def coefficient_of_variation_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: correlation coefficient

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['Q_var_rec'] = arg_df[arg_recorded.name] - arg_df[arg_recorded.name].mean()
    arg_df['Q_var_pre'] = arg_df[arg_predicted.name] - arg_df[arg_predicted.name].mean()
    arg_df['Q_sqr_var_rec'] = arg_df.Q_var_rec**2
    arg_df['Q_sqr_var_pre'] = arg_df.Q_var_pre**2
    arg_df['Q_prod_var_rec_pre'] = arg_df.Q_var_rec*arg_df.Q_var_pre
    
    # correlation coefficient (R2)
    r2 = (arg_df.Q_prod_var_rec_pre.sum()**2)/(arg_df.Q_sqr_var_rec.sum()*arg_df.Q_sqr_var_pre.sum())

    arg_df.to_csv(path_or_buf='{}df__{}__R2__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return r2, arg_df

# %%
def volume_variation_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: volume difference

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['delta_time'] = arg_df.index.to_series().diff()/pd.Timedelta(seconds=1)
    arg_df['Q_rec_lft'] = arg_df[arg_recorded.name].shift(periods=1)
    arg_df['Q_rec_rgt'] = arg_df[arg_recorded.name].shift(periods=-1).shift(periods=1)
    arg_df['Q_pre_lft'] = arg_df[arg_predicted.name].shift(periods=1)
    arg_df['Q_pre_rgt'] = arg_df[arg_predicted.name].shift(periods=-1).shift(periods=1)
    arg_df['Q_ave_rec'] = (arg_df.Q_rec_lft + arg_df.Q_rec_rgt)*arg_df.delta_time/2
    arg_df['Q_ave_pre'] = (arg_df.Q_pre_lft + arg_df.Q_pre_rgt)*arg_df.delta_time/2

    # recorded average volume
    vol_rec = arg_df.Q_ave_rec.sum()
    # simulated average volume
    vol_pre = arg_df.Q_ave_pre.sum()
    # volume variation (VV)
    vv = vol_pre - vol_rec

    arg_df.to_csv(path_or_buf='{}df__{}__VV__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return vv, arg_df

# %%
def volume_variation_relative_metric(arg_recorded, arg_predicted, arg_output_dir, arg_event='event'):
    # other name: volume difference

    arg_df = pd.concat(objs=[arg_recorded, arg_predicted], axis=1)

    arg_df['delta_time'] = arg_df.index.to_series().diff()/pd.Timedelta(seconds=1)
    arg_df['Q_rec_lft'] = arg_df[arg_recorded.name].shift(periods=1)
    arg_df['Q_rec_rgt'] = arg_df[arg_recorded.name].shift(periods=-1).shift(periods=1)
    arg_df['Q_pre_lft'] = arg_df[arg_predicted.name].shift(periods=1)
    arg_df['Q_pre_rgt'] = arg_df[arg_predicted.name].shift(periods=-1).shift(periods=1)
    arg_df['Q_ave_rec'] = (arg_df.Q_rec_lft + arg_df.Q_rec_rgt)*arg_df.delta_time/2
    arg_df['Q_ave_pre'] = (arg_df.Q_pre_lft + arg_df.Q_pre_rgt)*arg_df.delta_time/2

    # recorded average volume
    vol_rec = arg_df.Q_ave_rec.sum()
    # simulated average volume
    vol_pre = arg_df.Q_ave_pre.sum()
    # volume variation (VV)
    vv = (vol_pre - vol_rec)/vol_rec

    arg_df.to_csv(path_or_buf='{}df__{}__VV_rel__{}__{}.csv'.format(arg_output_dir, arg_event, arg_recorded.name, arg_predicted.name))
    
    return vv, arg_df

# %% [markdown]
# # Calculate metrics

# %%
data

# %%
def calculating_metrics(arg_name, arg_df, arg_output_dir):
    
    rec_flow = arg_df.columns[0]
    pre_flow = arg_df.columns[1:]

    func = [
        relative_error_metric, absolute_error_metric, time_to_peak_metric, time_to_centroid_metric,
        sum_of_absolute_difference_abs_metric, sum_of_absolute_difference_rel_metric,
        sum_of_the_squared_error_abs_metric, sum_of_the_squared_error_rel_metric,
        mean_squared_error_metric, root_mean_squared_error_metric, mean_error_metric, 
        mean_absolute_error_metric, percent_bias_metric,
        error_variance_metric, nash_sutcliffe_efficiency_metric, kling_gupta_efficiency_metric,
        coefficient_of_variation_metric, volume_variation_metric, volume_variation_relative_metric
        ]

    inds = ['simulation_{:02}'.format(ind + 1) for ind in range(len(pre_flow))]

    cols = [
        'relative_error', 'absolute_error', 'time_to_peak', 'time_to_centroid',
        'sum_of_absolute_difference_abs', 'sum_of_absolute_difference_rel',
        'sum_of_the_squared_error_abs', 'sum_of_the_squared_error_rel',
        'mean_squared_error', 'root_mean_squared_error', 'mean_error',
        'mean_absolute_error', 'percent_bias', 
        'error_variance', 'nash_sutcliffe_efficiency', 'kling_gupta_efficiency',
        'coefficient_of_variation', 'volume_variation', 'volume_variation_relative'
        ]

    metric = []
    df = []

    for ind1 in pre_flow:
        metric_flow = []
        df_flow = []

        for ind2 in func:
            metric_val, df_val = ind2(arg_df[rec_flow], arg_df[ind1], arg_output_dir, arg_name)
            metric_flow.append(metric_val)
            df_flow.append(df_val)

        metric.append(metric_flow)
        df.append(df_flow)
    
    metric = pd.DataFrame(data=metric, columns=cols, index=inds)
    df = pd.DataFrame(data=df, columns=cols, index=inds)

    return metric, df

# %%
data[['metric', 'df']] = data.apply(func=lambda arg: calculating_metrics(arg.sheet, arg.flow_rate, arg_output_dir), axis=1, result_type='expand')
data

# %%
def add_col_index(arg_event, arg_data):
    arg_data.index.name = 'simulation'
    arg_data.reset_index(inplace=True)
    arg_data = pd.concat(objs=[pd.Series(data=[arg_event]*len(arg_data), name='event'), arg_data], axis=1)

    return arg_data

# %%
data.metric = data.apply(func=lambda arg: add_col_index(arg.sheet, arg.metric), axis=1)
data

# %%
data.df = data.apply(func=lambda arg: add_col_index(arg.sheet, arg.df), axis=1)
data

# %%
metric_data = pd.concat(objs=data.metric.to_list())
metric_data.reset_index(drop=True, inplace=True)
metric_data

# %%
metric_data.to_csv(path_or_buf='{}metric_table.csv'.format(arg_output_dir), index=False)

# %%
tables_data = pd.concat(objs=data.df.to_list())
tables_data.reset_index(drop=True, inplace=True)
tables_data

# %%
def merge_tables(arg_df):
    arg_df = pd.concat(objs=[ind for ind in arg_df], axis=1)
    arg_df = arg_df.loc[:,~arg_df.columns.duplicated()]

    return arg_df

# %%
tables_data['other_tables'] = tables_data.iloc[:,6:].apply(func=lambda arg: merge_tables(arg), axis=1)
tables_data.drop(columns=tables_data.iloc[:,6:-1], inplace=True)
tables_data

# %%
def save_table_to_csv(arg_event, arg_simulation, arg_df, arg_output_dir):
    arg_df.to_csv(path_or_buf='{}tables__{}__{}.csv'.format(arg_output_dir, arg_event, arg_simulation))

# %%
tables_data.apply(func=lambda arg: save_table_to_csv(arg.event, arg.simulation, arg.other_tables, arg_output_dir), axis=1)

# %%
def rounding(n, decimal=0):
    if type(n) == pd.Timedelta:
        return n.round(freq='1s')
    else:
        multp = 10**decimal
        n_trunc = math.trunc(n*multp)/multp
        if math.trunc((n - n_trunc)*(multp*10)) >= 5:
            return math.trunc((n_trunc + 1/multp)*multp)/multp
        else:
            return n_trunc

# %%
def index_from_point(arg_array_df_series):
    """return an array containing the first pair-wise indices of a dataframe/series

    arguments:
        [array/dataframe/series] --> arg_array_df_series = any array with rows > 0
    """

    from_i = np.empty(shape=0, dtype=np.int64)

    for row_i in np.arange(stop=arg_array_df_series.shape[0] - 1):
        temp = np.full(shape=arg_array_df_series.shape[0] - 1 - row_i, fill_value=row_i)
        from_i = np.append(from_i, temp)

    return from_i

# %%
def index_to_point(arg_array_df_series):
    """return an array containing the second pair-wise indices of a dataframe/series

    arguments:
        [array/dataframe/series] --> arg_array_df_series = any array with rows > 0
    """
    
    to_j = np.empty(shape=0, dtype=np.int64)

    for row_j in np.arange(stop=arg_array_df_series.shape[0]):
        temp = np.arange(start=row_j + 1, stop=arg_array_df_series.shape[0])
        to_j = np.append(to_j, temp)

    return to_j

# %%
metric_columns = metric_data.columns[2:].to_numpy()
metric_columns

# %%
metric_tags = np.array(object=[
    'RE', 'AE', 'TtoP', 'TtoC', 'SofAD_abs', 'SofAD_rel', 'SoftheSE_abs',
    'SoftheSE_rel', 'MSE', 'RMSE', 'ME', 'MAE', 'PBIAS', 'S2', 'NSE', 'KGE', 'R2', 'VV', 'VV_rel'
    ])
metric_tags

# %%
col_i = metric_columns[index_from_point(metric_columns)]
col_i

# %%
col_j = metric_columns[index_to_point(metric_columns)]
col_j

# %%
tag_i = metric_tags[index_from_point(metric_tags)]
tag_i

# %%
tag_j = metric_tags[index_to_point(metric_tags)]
tag_j

# %%
metric_data.set_index(keys=['event', 'simulation'], inplace=True)
metric_data

# %%
metric_data = pd.DataFrame(data=metric_data.groupby(by='event'), columns=['event', 'data'])
metric_data

# %%
def create_save_fig(arg_output_dir, arg_event, arg_df, arg_col_i, arg_col_j, arg_tag_i, arg_tag_j):
    fig, ax = plt.subplots(figsize=(15,8))

    ax = arg_df.plot(
        x=arg_col_i,
        y=arg_col_j,
        kind='scatter',
        ax=ax,
        grid=True
    )

    ax.set_title(label='{} vs {}\n{}'.format(
        arg_col_j,
        arg_col_i,
        arg_event
        ))

    for ind in arg_df.index:
        ax.annotate(
            text='{}\n({}, {})'.format(
                ind[1],
                rounding(arg_df.loc[ind][arg_col_i],3),
                rounding(arg_df.loc[ind][arg_col_j],3)
                ), 
            xy=(
                arg_df.loc[ind][arg_col_i],
                arg_df.loc[ind][arg_col_j]
                ),
            xytext=(5,-10),
            textcoords='offset points'
            )
        
    fig.savefig(fname='{}fig__{}__{}_vs_{}.png'.format(
        arg_output_dir,
        arg_event,
        arg_tag_i,
        arg_tag_j
        ))

# %%
def pass_create_save_fig(arg_event, arg_df, arg_col_i, arg_col_j, arg_tag_i, arg_tag_j, arg_output_dir):
    arg_df['time_to_peak'] = arg_df['time_to_peak']/pd.Timedelta(minutes=1)
    arg_df['time_to_centroid'] = arg_df['time_to_centroid']/pd.Timedelta(minutes=1)
    
    for ind1, ind2, ind3, ind4 in zip(arg_col_i, arg_col_j, arg_tag_i, arg_tag_j):
        df = pd.concat(objs=[arg_df[ind1], arg_df[ind2]], axis=1)
        create_save_fig(arg_output_dir, arg_event, df, ind1, ind2, ind3, ind4)

# %%
metric_data.apply(func=lambda arg: pass_create_save_fig(arg.event, arg.data, col_i, col_j, tag_i, tag_j, arg_output_dir), axis=1)

# %%
zero_best_metric_from_zero_to_inf = [
    'relative_error', 'absolute_error', 'time_to_peak', 'time_to_centroid',
    'sum_of_absolute_difference_abs', 'sum_of_absolute_difference_rel',
    'sum_of_the_squared_error_abs', 'sum_of_the_squared_error_rel',
    'mean_squared_error', 'root_mean_squared_error',
    'mean_absolute_error', 'error_variance'
    ]
zero_best_metric_from_neg_inf_to_inf = [
    'mean_error',
    'percent_bias', 
    'volume_variation',
    'volume_variation_relative'
    ]
one_best_metric_from_neg_inf_or_zero_to_one = [
    'nash_sutcliffe_efficiency', 'kling_gupta_efficiency',
    'coefficient_of_variation'
    ]

conditions_ranking = [
    zero_best_metric_from_zero_to_inf,
    zero_best_metric_from_neg_inf_to_inf,
    one_best_metric_from_neg_inf_or_zero_to_one
    ]

print(
    *conditions_ranking,
    sep='\n'
    )

# %%
def ranking_metric_series(arg_df, arg_conditions=conditions_ranking):
    zero_best_metric_from_zero_to_inf = arg_conditions[0]
    zero_best_metric_from_neg_inf_to_inf = arg_conditions[1]
    one_best_metric_from_neg_inf_or_zero_to_one = arg_conditions[2]

    for col in arg_df.columns:
        temp_series = arg_df[col]
        if col in zero_best_metric_from_zero_to_inf:
            rank_serie = temp_series.abs().rank(ascending=True)
            arg_df[col] = rank_serie
        elif col in zero_best_metric_from_neg_inf_to_inf:
            rank_serie = temp_series.abs().rank(ascending=True)
            arg_df[col] = rank_serie
        elif col in one_best_metric_from_neg_inf_or_zero_to_one:
            rank_serie = temp_series.abs().rank(ascending=False)
            arg_df[col] = rank_serie
    
    return arg_df

# %%
metric_ranking = metric_data.copy()
metric_ranking = pd.concat(objs=[ind for ind in metric_ranking.apply(func=lambda arg: ranking_metric_series(arg.data), axis=1)])
metric_ranking['average'] = metric_ranking.mean(axis=1, numeric_only=True)
metric_ranking

# %%
metric_ranking.to_csv(path_or_buf='{}metric_ranking.csv'.format(arg_output_dir))

# %%



