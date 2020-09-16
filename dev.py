import datetime
import os

import imageio
import ee
import yaml

from data_preparation.DatasetPrepareService import DatasetPrepareService
from data_preparation.utils.EarthEngineMapClient import EarthEngineMapClient

with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    ee.Initialize()
    satellites = ['GOES', 'MODIS', 'VIIRS', 'Sentinel2']
    locations = ['LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex',
                 'August_complex', 'Beachie_wildfire', 'Beachie_wildfire_2', 'Holiday_farm_wildfire',
                 'North_complex_fire', 'North_complex_fire_2', 'North_complex_fire_3',
                 'Cold_spring_fire', 'Doctor_creek_fire']
    enable_visualization = True
    enable_downloading = False
    generate_gif_for_goes = True
    custom_generate_per_day = True
    for location in locations:
        start_time = config.get(location).get('start')
        # end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        filenames = []
        map_client = EarthEngineMapClient(location)

        dataset_pre = DatasetPrepareService(location=location)
        time_dif = end_time - start_time

        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                img_collection, img_collection_as_gif = dataset_pre.prepare_daily_image(custom_generate_per_day, satellite=satellite, date_of_interest=date_of_interest)
                img_to_visualize = img_collection.max()
                if len(img_to_visualize.getInfo().get('bands')) != 0:
                    if enable_downloading:
                        dataset_pre.download_image_to_gcloud(img_to_visualize.toFloat(), satellite, date_of_interest)
                if enable_visualization:
                    map_client = dataset_pre.visualize_in_openstreetmap(img_collection.max(), map_client, satellite, date_of_interest)
                if generate_gif_for_goes and satellite == 'GOES':
                    dataset_pre.download_collection_as_video(img_collection_as_gif, satellite, date_of_interest)
                if os.path.isfile('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    filenames.append(satellite + str(date_of_interest))
        if enable_visualization:
            map_client.initialize_map()

        if custom_generate_per_day:
            images = []
            for filename in filenames:
                images.append(imageio.imread('images_for_gif/' + location + '/' + filename + '.jpg'))
            imageio.mimsave('images_for_gif/'+ location + '.gif', images, format='GIF', fps=1)
