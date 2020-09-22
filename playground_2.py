import datetime
import os
import cv2
import ee
import imageio
import yaml

from data_preparation.DatasetPrepareService import DatasetPrepareService

with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ee.Initialize()
if __name__ == '__main__':
    satellites = ['Sentinel2']
    locations = ['creek_fire']
    for location in locations:
        start_time = config.get(location).get('start')
        #end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        time_dif = end_time - start_time
        dataset_pre = DatasetPrepareService(location=location)
        custom_generate_per_day = False
        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                if os.path.isfile('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    geomsample = dataset_pre.prepare_image_patch(satellite=satellite, date_of_interest=date_of_interest)