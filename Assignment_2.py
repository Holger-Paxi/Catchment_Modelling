# %% [markdown]
# A catchment modelling system for simulation of flows in the Powells Creek Stormwater System has been developed using the Stormwater Management Model (SWMM).  The Powells Creek stormwater system is located near the Olympic Park area in the inner West of Sydney.
# 
# The ultimate aim of the catchment modelling system is the reproduction of both the quantity and quality of stormwater from the upstream catchment.  To increase the usefulness of the model, it is being calibrated against recorded data from a gauging station (located at Elva Ave, Strathfield) operated by the School of Civil and Environmental Engineering, UNSW.
# 
# An intern has created a reproduction of the recorded event. As their supervisor you are required to review their work and provide comment on how to improve the calibration achieved to date. Your manager has requested that you present your findings to them and provide them with a 500 word briefing note (or memo) to summarise the findings and future actions that they can share with the project manager and client.
# 
# The intern has provided and collated the following data is provided
# 
# * Recorded levels at the Elva Avenue Gauging Station for an event that occurred on 23 April 1989.
# * The rating table for the Elva Avenue Gauging Station.
# * A plot of the rating table for the Elva Avenue Gauging Station.
# * The output for a simulation of the 23 April 1989 event using SWMM.
# 
# `Strathfield Recorded Levels 213304` download
# 
# `Strathfield rating table 213304` download
# 
# `Strathfield Stage Discharge 213304` download
# 
# `Simulation Results` download
# 
# The intern also prepared the `Bulk download of files` (download) for your convenience.
# 
# To assist, the intern has provided:
# 
# * Spreadsheet with the data in an excel format.
# * Video explaining the data that they found online.
# 
# `Strathfield Project 1989-2020.xlsx`  download

# %%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os

# %% [markdown]
# # read input data

# %% [markdown]
# ## simulated data

# %%
sim_data = pd.read_excel(io='./Inputs_CM_Assignment_2/strathfield 1989 2020.xlsx', sheet_name='Predicted Flows')
sim_data.drop(index=[136,137], inplace=True)
sim_data['Timestamp'] = pd.to_datetime(arg=sim_data.Date) + pd.to_timedelta(arg=sim_data.Time.astype(str))
sim_data = sim_data[sim_data.columns[[3,2]]]
sim_data

# %% [markdown]
# ## recorded data

# %%
rec_data = pd.read_excel(io='./Inputs_CM_Assignment_2/strathfield 1989 2020.xlsx', sheet_name='Recorded Flows')
rec_data['Timestamp'] = pd.to_datetime(arg=rec_data.Date) + pd.to_timedelta(arg=rec_data.Time.astype(str))
rec_data = rec_data[rec_data.columns[[4,3]]]
rec_data

# %% [markdown]
# ## rating curve

# %%
rat_cur = pd.read_table(filepath_or_buffer='./Inputs_CM_Assignment_2/strathfield/strathfield.txt', sep='\s+', skiprows=10)
rat_cur.drop(index=range(37,43), inplace=True)
rat_cur.drop(index=range(47,50), inplace=True)
rat_cur.reset_index(drop=True, inplace=True)
rat_cur.set_index(keys='G.H.', inplace=True)
rat_cur.astype(float)
rat_cur

# %%
def rounding(n, decimal=0):
    multp = 10**decimal
    n_trunc = math.trunc(n*multp)/multp
    if math.trunc((n - n_trunc)*(multp*10)) >= 5:
        return math.trunc((n_trunc + 1/multp)*multp)/multp
    else:
        return n_trunc

# %%
info = np.array([
    [rounding(float(indc) + float(indr), 2), float(rat_cur.loc[indr, indc])] 
    for indr in rat_cur.index 
    for indc in rat_cur.columns])
info

# %%
rat_cur = pd.DataFrame(data=info, columns=['Depth', 'Flow'])
rat_cur.dropna(axis=0, inplace=True)
rat_cur.reset_index(drop=True, inplace=True)
rat_cur

# %%
new_ind = pd.DataFrame(data=np.arange(start=rat_cur.Depth.min(), stop=rat_cur.Depth.max(), step=0.001), columns=['Depth'])
new_ind.Depth.apply(lambda arg: rounding(arg, 3))
new_ind

# %%
rat_cur = pd.concat(objs=[rat_cur, new_ind], axis=0)
rat_cur.Depth = rat_cur.Depth.apply(lambda arg: rounding(arg, 3))
rat_cur.drop_duplicates(subset='Depth', inplace=True)
rat_cur.sort_values(by='Depth', inplace=True)
rat_cur.reset_index(drop=True, inplace=True)
rat_cur.interpolate(method='linear', axis=0, inplace=True)
rat_cur

# %% [markdown]
# # merging recorded data and rating curve

# %% [markdown]
# ## recorded data

# %%
rec_data = pd.merge(left=rat_cur, right=rec_data, how='right', on='Depth')
rec_data = rec_data[rec_data.columns[[2,1]]]
rec_data

# %%
print('min: {}\nmax: {}'.format(rec_data.Timestamp.min(), rec_data.Timestamp.max()))
print('min: {}\nmax: {}'.format(rec_data.Flow.min(), rec_data.Flow.max()))

# %%
print('min: {}\nmax: {}'.format(sim_data.Timestamp.min(), sim_data.Timestamp.max()))
print('min: {}\nmax: {}'.format(sim_data.Flow.min(), sim_data.Flow.max()))

# %% [markdown]
# # create output directory

# %%
arg_output_dir = './Outputs_CM_Assignment_2/'
arg_output_dir

# %%
def create_output_dir(arg_output_dir):
    """create output directory if it does not exist

    arguments:
        arg_output_dir = [string] './Outputs_CM_Assignment_2/'
    """
    if not os.path.exists(arg_output_dir):
        os.makedirs(arg_output_dir)

# %% [markdown]
# # comparing simulated and recorded data

# %%
def move_compare_hydrographs(minutes=0, sim_data=sim_data, rec_data=rec_data):
    # delta time
    delta_time = dt.timedelta(minutes=minutes)
    
    # transforming data by delta time
    sim_data_transformed = sim_data.copy()
    sim_data_transformed.Timestamp = sim_data_transformed.Timestamp + delta_time
    
    # minimum and maximum timestamps
    min_time = sim_data_transformed.Timestamp.min()
    max_time = sim_data_transformed.Timestamp.max()

    # dataframe
    df = pd.concat(objs=[rec_data.set_index(keys='Timestamp'), sim_data_transformed.set_index(keys='Timestamp')], axis=1)
    df.reset_index(drop=False, inplace=True)
    df = df[(df.Timestamp >= min_time) & (df.Timestamp <= max_time)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = ['Timestamp', 'Recorded_data', 'Simulated_data']

    # columns for metrics
    df['dif'] = df.Simulated_data - df.Recorded_data
    df['abs_dif'] = df.dif.abs()
    df['sq_dif'] = df.dif**2
    df['rel_sq_dif'] = (df.dif/df.Recorded_data)**2
    df['dif_rec'] = df.Recorded_data - df.Recorded_data.mean()
    df['dif_sim'] = df.Simulated_data - df.Simulated_data.mean()
    df['sq_dif_rec'] = df.dif_rec**2
    df['sq_dif_sim'] = df.dif_sim**2
    df['prod_rec_sim'] = df.dif_rec*df.dif_sim
    df['delta_time'] = df.Timestamp.diff()/dt.timedelta(seconds=1)
    df['rec_lft'] = df.Recorded_data.shift(periods=1)
    df['rec_rgt'] = df.Recorded_data.shift(periods=-1).shift(periods=1)
    df['sim_lft'] = df.Simulated_data.shift(periods=1)
    df['sim_rgt'] = df.Simulated_data.shift(periods=-1).shift(periods=1)
    df['ave_rec'] = (df.rec_lft + df.rec_rgt)*df.delta_time/2
    df['ave_sim'] = (df.sim_lft + df.sim_rgt)*df.delta_time/2

    # root mean square error
    rmse = df.sq_dif.mean()**0.5
    # mean squared error
    mse = df.sq_dif.mean()
    # sum of square differences - absolute
    ssd_abs = df.sq_dif.sum()
    # sum of square differences - relative
    ssd_rel = df.rel_sq_dif.sum()
    # error variance
    s2 = ssd_abs/(len(df.index) - 1)
    # Nash-Sutcliffe Efficiency - modelling efficiency
    nse = 1 - (df.sq_dif.sum()/df.sq_dif_rec.sum())
    # mean error - mean bias error
    me = df.dif.mean()
    # mean absolute error
    mae = df.abs_dif.mean()
    # percent bias
    pbias = -df.dif.sum()*100/df.Recorded_data.sum()
    # coefficient of variation
    r2 = (df.prod_rec_sim.sum()**2)/(df.sq_dif_rec.sum()*df.sq_dif_sim.sum())
    # Kling-Gupta Efficacy
    kge = (
        1 - (
            (((r2**0.5) - 1)**2) + 
            (((df.Simulated_data.mean()/df.Recorded_data.mean()) - 1)**2) + 
            ((((df.Simulated_data.std()/df.Simulated_data.mean())/(df.Recorded_data.std()/df.Recorded_data.mean())) - 1)**2)
        )**0.5
    )
    # recorded average volume
    vol_rec = df.ave_rec.sum()
    # simulated average volume
    vol_sim = df.ave_sim.sum()
    # volume difference
    vol_dif = vol_sim - vol_rec
    # relative volume difference
    vol_dif_rel = (vol_sim - vol_rec)/vol_rec

    # dataframe metrics
    metrics = pd.DataFrame(data=np.array([
        ['rmse', rmse], 
        ['mse', mse], 
        ['ssd_abs', ssd_abs], 
        ['ssd_rel', ssd_rel], 
        ['s2', s2],
        ['nse', nse],
        ['me', me],
        ['mae', mae],
        ['pbias', pbias],
        ['r2', r2],
        ['kge', kge],
        ['vol_dif', vol_dif],
        ['vol_dif_rel', vol_dif_rel]
        ]), columns=['metric', 'value'])

    # plot
    fig, ax = plt.subplots()
    ax = df.plot(
        x='Timestamp', 
        y=['Recorded_data', 'Simulated_data'], 
        kind='line',
        ax=ax, 
        figsize=(15,8), 
        title='Recorded Data and Simulated Data\n{}-minute shifted'.format(int(minutes)),
        grid=True,
        xlabel='Timestamp',
        ylabel='Flow Rate in (m^3/s)'
        )
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d %H:%M'))

    # save figures
    fig.savefig(arg_output_dir + 'Rec_Data_and_Sim_Data_{:>03}_min.png'.format(int(minutes)))

    return delta_time, sim_data_transformed, min_time, max_time, df, metrics

# %%
proposed_trans_values = pd.DataFrame(data=np.arange(start=0, stop=120, step=5), columns=['transformation'])
proposed_trans_values

# %%
proposed_trans_values = proposed_trans_values.apply(lambda arg: move_compare_hydrographs(float(arg.transformation)), axis=1, result_type='expand')
proposed_trans_values.columns = ['delta_time', 'sim_data_transformed', 'min_time', 'max_time', 'df', 'metrics']
proposed_trans_values

# %% [markdown]
# # export results to csv

# %%
for ind in proposed_trans_values.index:
    proposed_trans_values.df[ind].reset_index(drop=True).to_csv(
        path_or_buf=arg_output_dir + 'shift_{:>03}_min.csv'.format(int(proposed_trans_values.delta_time[ind]/dt.timedelta(minutes=1))), 
        index=False
        )

# %% [markdown]
# # calibration metrics and exporting results

# %%
calib_metric = pd.concat(objs=[ind.set_index(keys='metric') for ind in proposed_trans_values.metrics], axis=1)
calib_metric.columns = ['shift_{:>02}_min'.format(int(ind/dt.timedelta(minutes=1))) for ind in proposed_trans_values.delta_time]
calib_metric.reset_index(inplace=True)
calib_metric.set_index(keys='metric', inplace=True)
calib_metric = calib_metric.T
calib_metric = calib_metric.apply(pd.to_numeric)
calib_metric.reset_index(inplace=True)
calib_metric.rename(columns={'index':'shift_time'}, inplace=True)
calib_metric.columns.name = None
calib_metric.to_csv(path_or_buf=arg_output_dir + 'calibration_metric.csv', index=False)
calib_metric.set_index(keys='shift_time', inplace=True)
calib_metric

# %% [markdown]
# # figures and exporting figures

# %%
for col in calib_metric.columns:
    fig, ax = plt.subplots()
    ax = calib_metric.plot(
        y='{}'.format(col),
        kind='bar',
        ax=ax,
        figsize=(15,8),
        title='{} values for different shifted hydrographs'.format(calib_metric[col].name.upper()),
        ylabel='{}'.format(calib_metric[col].name.upper())
        )
    fig.savefig(arg_output_dir + 'bar_chart_{}.png'.format(calib_metric[col].name.upper()))

# %%



