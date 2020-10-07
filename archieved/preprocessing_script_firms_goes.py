import datetime
import os

import numpy as np
import ee
import yaml
import rasterio

from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Preprocessing.PreprocessingService import PreprocessingService

with open("DataPreparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ee.Initialize()

preprocessing = PreprocessingService()

if __name__ == '__main__':
    satellites = ['GOES', 'FIRMS']
    locations = ['LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex', 'creek_fire', 'August_complex']
    for location in locations:
        start_time = config.get(location).get('start')
        end_time = config.get(location).get('end')
        #end_time = datetime.date.today()
        time_dif = end_time - start_time
        dataset_pre = DatasetPrepareService(location=location)
        custom_generate_per_day = False
        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            firms_arr_source, firms_profile = preprocessing.read_tiff('label/' + location + 'FIRMS' + '/' + "Cal_fire_" + location + 'FIRMS' + '-' + str(date_of_interest) + '.tif')
            goes_arr, goes_profile = preprocessing.read_tiff('data/' + location + 'GOES' + '/' + "Cal_fire_" + location + 'GOES' + '-' + str(date_of_interest) + '.tif')
            firms_arr = preprocessing.padding(goes_arr, np.nan_to_num(firms_arr_source, 0.0))
            x_size = goes_arr.shape[1]
            y_size = goes_arr.shape[2]

            downsampled_firms = np.zeros((1, x_size, y_size))
            for i in range(x_size):
                for j in range(y_size):
                    downsampled_firms[0, i, j] = preprocessing.down_sampling(firms_arr[0, 2 * i: 2 * i + 2, 2 * j: 2 * j + 2])

            destination_name = 'label/' + location + ' label' + '/' + "Cal_fire_" + location + 'GOES' + '-' + str(date_of_interest) + '.tif'
            dir_name = os.path.dirname(destination_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            goes_profile.update(count=1)
            preprocessing.write_tiff(destination_name, downsampled_firms, goes_profile)