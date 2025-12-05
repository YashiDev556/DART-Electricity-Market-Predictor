import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rt_xls = pd.ExcelFile('Data/rpt.00013061.0000000000000000.RTMLZHBSPP_2011.xlsx')
print(rt_xls.sheet_names)

rt = pd.concat(
    (pd.read_excel(rt_xls, sheet_name=s) for s in rt_xls.sheet_names),
    ignore_index=True
)

zone = "LZ_HOUSTON"

rtm_z = rt[rt["Settlement Point Name"] == zone].copy()



mask_24 = rtm_z["Delivery Hour"] == 24
rtm_z.loc[mask_24, "Delivery Hour"] = 0
rtm_z.loc[mask_24, "Delivery Date"] = (
    pd.to_datetime(rtm_z.loc[mask_24, "Delivery Date"]) + pd.Timedelta(days=1)
).dt.strftime("%m/%d/%Y")

rtm_z["datetime"] = pd.to_datetime(
    rtm_z["Delivery Date"] + " " + rtm_z["Delivery Hour"].astype(int).astype(str),
    format="%m/%d/%Y %H"
)

rtm_z = rtm_z.set_index("datetime").sort_index()
rtm_z = rtm_z.rename(columns={"Settlement Point Price": "RTM_15min"})

rtm_hourly = rtm_z["RTM_15min"].groupby("datetime").mean().to_frame("RTM")


fed_funds_df = pd.read_csv('Data/federal_funds_2011.csv')


fed_funds_rate = fed_funds_df


# columns: ['observation_date', 'FEDFUNDS']
fed_funds_rate['observation_date'] = pd.to_datetime(fed_funds_rate['observation_date'])
ffr_df = fed_funds_rate.set_index('observation_date').sort_index()

# start at first day of first month
start = ffr_df.index.min().normalize()               # 2011-01-01 00:00

# end = last day of last month, 23:00
last_month_end = ffr_df.index.max().to_period('M').to_timestamp('M')  # 2011-12-31 00:00
end = last_month_end + pd.Timedelta(hours=23)                          # 2011-12-31 23:00

# hourly index for the full year
hourly_idx = pd.date_range(start, end, freq='H')

# reindex and forward-fill monthly values
ffr_hourly = ffr_df.reindex(hourly_idx, method='ffill')

print(ffr_hourly.index[0], ffr_hourly.index[-1])
# 2011-01-01 00:00:00 2011-12-31 23:00:00
print(ffr_hourly.shape)

ffr_hourly = ffr_hourly[:-1]


import netCDF4

temp = netCDF4.Dataset('Data/1yeartexasfocused/data_stream-oper_stepType-instant.nc')

cloud = netCDF4.Dataset('Data/1yeartexasfocused/6d54a8966bd3e929ebba9c39b86323fe.nc')

wind = netCDF4.Dataset('Data/1yeartexasfocused/b19d9b13a745c2fbd15c421befa61f76.nc')


lat = np.asarray(temp.variables['latitude'])
lon = np.asarray(temp.variables['longitude'])
# temperature 2 meters above ground level
t2m = np.asarray(temp.variables['t2m']) # shape (time, lat, lon)

cloud_cover = np.asarray(cloud.variables['tcc']) # shape (time, lat, lon)
v_component = np.asarray(wind.variables['v10']) # shape (time, lat, lon)
u_component = np.asarray(wind.variables['u10']) # shape (time, lat, lon)



# time is in hours (31 * 24 = 744 hours)
t2m = t2m[1:, :, :]  # last 744 hours of December 2010
cloud_cover = cloud_cover[1:, :, :]
v_component = v_component[1:, :, :]
u_component = u_component[1:, :, :]
print(t2m.shape)
print(cloud_cover.shape)
print(v_component.shape)
print(u_component.shape) 


import numpy as np

# assuming u_component and v_component are already time × lat × lon
wind_speed = np.sqrt(u_component**2 + v_component**2)

print(wind_speed.shape)  # (time, lat, lon) same as u/v


# # taking the sub grid of interest (around Texas)
# temp_sub = t2m[:, 208:257, 1006:1064]
# temp_sub.shape

temp_sub = t2m
print(temp_sub.shape)

import yfinance as yf

data = yf.download(
    "SPY",
    start="2011-01-01",
    end="2012-01-01",
    interval="1d"
)





# 2. Convert daily SPY data to hourly array of length 744


spy_daily = data["Close"]

idx = pd.date_range("2010-12-01", "2010-12-31 23:00", freq="H")


spy_daily_full = spy_daily.asfreq("D").ffill()
idx = pd.date_range("2011-01-01", "2011-12-31", freq="D")
spy_daily_full = spy_daily.reindex(idx).ffill()


start = spy_daily_full.index.min()
end   = spy_daily_full.index.max()

idx_hourly = pd.date_range(start, end + pd.Timedelta(hours=23), freq="H")

spy_hourly = spy_daily_full.reindex(idx_hourly, method="ffill")[:-1]

print(rtm_hourly.shape)
print(spy_hourly.shape)

print(temp_sub.shape)
print(ffr_hourly.shape)



import numpy as np
import pandas as pd

T = rtm_hourly.shape[0]  # 744 for one month
feature_cols = ["RT", "SPY", "FFR", "hour", "dow", "month"]

idx = pd.date_range("2010-12-01", periods=T, freq="H")  # adjust start

df_scalar = pd.DataFrame(index=idx)
df_scalar["RT"]   = rtm_hourly.squeeze()
df_scalar["SPY"]  = spy_hourly.squeeze()
df_scalar["FFR"]  = ffr_hourly.astype("float32")

# calendar features
df_scalar["hour"]  = df_scalar.index.hour
df_scalar["dow"]   = df_scalar.index.dayofweek
df_scalar["month"] = df_scalar.index.month



# grid_all = np.stack(
#     [temp_sub, cloud_cover, u_component, v_component],
#     axis=-1
# ).astype("float32")

grid_all = np.stack(
    [temp_sub, cloud_cover, wind_speed],
    axis=-1
).astype("float32")


print(grid_all.shape) 


# grid_mean = grid_all.mean(axis=(0, 1, 2), keepdims=True)  # (1,1,1,4)
# grid_std  = grid_all.std(axis=(0, 1, 2), keepdims=True) + 1e-6
# grid_all  = (grid_all - grid_mean) / grid_std

mask_valid = ~df_scalar[feature_cols + ["RT"]].isna().any(axis=1)
df_scalar = df_scalar[mask_valid]
grid_all = grid_all[mask_valid.to_numpy()]  # keep t2m in sync with time axis

past_hours = 24
future_hours = 24


def make_supervised_spatiotemporal(df_scalar, grid_all, past_hours, future_hours, feature_cols):
    # df_scalar: DataFrame (T, n_scalar)
    # t2m_grid:  array (T, H, W)
    values = df_scalar[feature_cols].to_numpy().astype("float32")  # (T, n_scalar)
    rt     = df_scalar["RT"].to_numpy().astype("float32")          # (T,)
    
    
    n_scalar   = values.shape[1]
    T, H, W, C = grid_all.shape
    
    X_grid_list = []
    X_seq_list  = []
    Y_list      = []

    # here we get samples by sliding a window over time (each step is a sample of data)
    
    for t in range(T - past_hours - future_hours + 1): # so we dont overshoot the time axis
        # temperature block
        grid_block = grid_all[t : t + past_hours]                # (past_hours, H, W)
        grid_block = grid_block[..., np.newaxis]                  # (past_hours, H, W, 1)
        
        # scalar time-series block
        seq_block = values[t : t + past_hours]                    # (past_hours, n_scalar)
        
        # future RT prices
        future_rt = rt[t + past_hours : t + past_hours + future_hours]  # (24,)
        
        X_grid_list.append(grid_block)
        X_seq_list.append(seq_block)
        Y_list.append(future_rt)
    
    X_grid = np.stack(X_grid_list, axis=0)   # (n_samples, past_hours, H, W, 1)
    X_seq = np.stack(X_seq_list, axis=0)    # (n_samples, past_hours, n_scalar)
    Y = np.stack(Y_list, axis=0)        # (n_samples, future_hours)
    
    return X_grid, X_seq, Y

X_grid, X_seq, Y = make_supervised_spatiotemporal(
    df_scalar, grid_all, past_hours, future_hours, feature_cols
)

print("X_grid:", X_grid.shape)  # (n_samples, 24, H, W, 1)
print("X_seq: ", X_seq.shape)   # (n_samples, 24, n_scalar)
print("Y:     ", Y.shape)       # (n_samples, 24)


#######################################################

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # in MB
)
    


from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization, GlobalAveragePooling2D, Flatten,
    Conv1D, GlobalAveragePooling1D, Dense, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

n_scalar = X_seq.shape[2]
H, W, C = X_grid.shape[2], X_grid.shape[3], X_grid.shape[4]

# spatio-temporal grid branch
grid_in = Input(shape=(past_hours, H, W, C), name="grid_input")
xg = ConvLSTM2D(
    filters=16,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    return_sequences=False
)(grid_in)
xg = BatchNormalization()(xg)
xg = Flatten()(xg)
#xg = GlobalAveragePooling2D()(xg)   # -> (batch, 16)

# time series branch
seq_in = Input(shape=(past_hours, n_scalar), name="seq_input")
xs = Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(seq_in)
xs = Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(xs)
xs = GlobalAveragePooling1D()(xs)   # -> (batch, 32)

# combine
x = Concatenate()([xg, xs])         # (batch, 48)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
out = Dense(future_hours, name="output")(x)   # 24-hour forecast

model = Model(inputs=[grid_in, seq_in], outputs=out)
model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
print(model.summary())


split = int(len(X_grid) * 0.8)
Xg_train, Xg_test = X_grid[:split], X_grid[split:]
Xs_train, Xs_test = X_seq[:split],  X_seq[split:]
Y_train,  Y_test  = Y[:split],      Y[split:]

history = model.fit(
    [Xg_train, Xs_train], Y_train,
    validation_data=([Xg_test, Xs_test], Y_test),
    epochs=8,
    batch_size=4,
    verbose=1
)

last_grid_block = grid_all[-past_hours:]              # (24, H, W)
last_grid_block = last_grid_block[..., np.newaxis]   # (24, H, W, 1)

last_seq_block = df_scalar[feature_cols].to_numpy().astype("float32")[-past_hours:]

Xg_last = np.expand_dims(last_grid_block, axis=0)    # (1, 24, H, W, 1)
Xs_last = np.expand_dims(last_seq_block,  axis=0)    # (1, 24, n_scalar)

next_24_forecast = model.predict([Xg_last, Xs_last])[0]  # shape (24,)
print("Next 24-hour forecast:", next_24_forecast)





rt_2012_jan = pd.read_excel('Data/rpt.00013060.0000000000000000.DAMLZHBSPP_2012.xlsx', sheet_name='Jan_1')


zone = "LZ_HOUSTON"
rtm_z = rt_2012_jan[rt_2012_jan["Settlement Point"] == zone].copy()
rtm_z


rtm_z["Hour Ending"].str.split(":").str[0].astype(int)


# 1. make Delivery Date a datetime
rtm_z["Delivery Date"] = pd.to_datetime(rtm_z["Delivery Date"], format="%m/%d/%Y")

# 2. find the 24:00 rows and shift them to next day at 00:00
mask_24 = rtm_z["Hour Ending"].str.strip().eq("24:00")

rtm_z.loc[mask_24, "Delivery Date"] = rtm_z.loc[mask_24, "Delivery Date"] + pd.Timedelta(days=1)
rtm_z.loc[mask_24, "Hour Ending"]   = "00:00"

# 3. build a proper datetime column
rtm_z["datetime"] = pd.to_datetime(
    rtm_z["Delivery Date"].dt.strftime("%m/%d/%Y") + " " + rtm_z["Hour Ending"],
    format="%m/%d/%Y %H:%M"
)

# now you can set as index and aggregate
rtm_z = rtm_z.set_index("datetime").sort_index()
rtm_z = rtm_z.rename(columns={"Settlement Point Price": "RTM_15min"})
rtm_hourly_true = rtm_z["RTM_15min"].groupby("datetime").mean().to_frame("RTM")



import matplotlib.pyplot as plt

plt.plot(range(24), next_24_forecast, label="Neural Net Model Forecast")
#plt.plot(range(24), y_last_pred[0], label="Random Forest Model Forecast")
plt.plot(range(24), rtm_hourly_true[:24]["RTM"], label="True RTM Prices")
plt.xlabel("Hour")
plt.ylabel("RTM Price ($/MWh)")
plt.title("Next 24-hour RTM Price Forecasts for LZ_HOUSTON (Jan 1, 2011)")
plt.legend()
plt.show()





