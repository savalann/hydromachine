import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def Data_Preprocessing(data, split_year, length_lookback=0, length_prediction=1, scale=False):
    final_result = {}
    station_list = data['station_id'].drop_duplicates()
    column_names = ['station_id', 'datetime', 'Lat', 'Long', 'Drainage_area_mi2', 'Perc_Develop',
                    'Perc_Imperv', 'Perc_Slop_30', 's1', 's2', 'storage', 'swe', 'NWM_flow',
                    'DOY', 'flow_cfs']
    data = data[column_names]
    final_result['column_names'] = data.columns
    data.reset_index(drop=True, inplace=True)

    if scale == True:
        data_temp_00 = data.drop(columns=['datetime', 'station_id']).reset_index(drop=True)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        data_temp_01 = scaler.fit_transform(data_temp_00)
        data_temp_01 = np.concatenate(
            (data['station_id'].to_numpy().reshape(-1, 1), data['datetime'].to_numpy().reshape(-1, 1), data_temp_01),
            axis=1)
        final_result['scaler'] = scaler
    else:
        data_temp_01 = data.reset_index(drop=True)

    if length_lookback == 0 and length_prediction == 1:
        data_temp_01 = data_temp_01.to_numpy()
        for data_type in ['train', 'test']:
            if data_type == 'train':
                row_data = data.datetime < f'01-01-{split_year}'
            if data_type == 'test':
                row_data = data.datetime >= f'01-01-{split_year}'
                final_result['test_station'] = data_temp_01[row_data][:, 0]
                final_result['test_datetime'] = data_temp_01[row_data][:, 1]

            final_result[f'x_{data_type}'] = np.delete(data_temp_01[row_data][:, :-1], [0, 1], axis=1)

            final_result[f'y_{data_type}'] = data_temp_01[row_data][:, -1]
    else:
        for data_type in ['train', 'test']:
            data_x_all, data_y_all = [], []
            for station_number in station_list:

                if data_type == 'train':
                    row_data = (data.station_id == station_number) & (data.datetime < f'01-01-{split_year}')
                if data_type == 'test':
                    row_data = (data.station_id == station_number) & (data.datetime >= f'01-01-{split_year}')

                data_temp_02 = data_temp_01[row_data]
                data_x, data_y = [], []
                for i in range(len(data_temp_02) - length_lookback + length_prediction - 1):
                    # find the end of this pattern
                    features, targets = data_temp_02[i:i + length_lookback, :-1], data_temp_02[
                                                                                  i + length_lookback:i + length_lookback + length_prediction,
                                                                                  -1]
                    data_x.append(features)
                    data_y.append(targets)
                data_x_all.extend(data_x)
                data_y_all.extend(data_y)
            if data_type == 'test':
                final_result['test_station'] = np.array(data_x_all)[:, 0, 0]

            final_result[f'x_{data_type}'] = torch.Tensor(np.delete(np.array(data_x_all).astype(np.float64), 0, axis=2))
            final_result[f'y_{data_type}'] = torch.Tensor(np.array(data_y_all).astype(np.float64))

    return final_result
