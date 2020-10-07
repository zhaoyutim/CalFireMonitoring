import datetime

import numpy as np
import pandas as pd
import yaml
import ee
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer

from Preprocessing.PreprocessingService import PreprocessingService
from DataPreparation.DatasetPrepareService import DatasetPrepareService

ee.Initialize()
with open("DataPreparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def convert_int_to_timestamp(number, period):
    mins = int(number % 100)
    hours = int(number / 100 % 100)
    start = "{:02d}:{:02d}".format(max(hours - period, 0), mins)
    end = "{:02d}:{:02d}".format(min(hours + period, 23), mins)
    return start, end


if __name__ == '__main__':
    locations = ['creek_fire']

    for location in locations:
        start_time = config.get(location).get('start')
        preprocessing = PreprocessingService()
        # end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        filenames = []
        time_dif = end_time - start_time
        dataset_pre = DatasetPrepareService(location=location)
        longitude = config.get(location).get('longitude')
        latitude = config.get(location).get('latitude')
        size = config.get('rectangular_size')
        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            goes_arr, goes_profile = preprocessing.read_tiff(
                'data/' + location + 'GOES' + '/' + location + 'GOES_' + "Cal_fire_" + location + 'GOES' + '-' + '2020-09-04.tif')
            bbox = [goes_profile.data.get('transform').column_vectors[2][0], goes_profile.data.get('transform').column_vectors[2][0]+goes_profile.data.get('transform').column_vectors[0][0]*goes_profile.data.get('width'),
                    goes_profile.data.get('transform').column_vectors[2][1]+goes_profile.data.get('transform').column_vectors[1][1]*goes_profile.data.get('height'), goes_profile.data.get('transform').column_vectors[2][1]]
            satellite='FIRMS'
            transformer = Transformer.from_crs(32610, 4326)

            # bbox = [latitude - size, latitude + size, longitude - size, longitude + size]
            bot_left = transformer.transform(bbox[0], bbox[2])
            top_right = transformer.transform(bbox[1], bbox[3])

            #  Initialize the Image Size

            #  Choose some Geographic Transform (Around Lake Tahoe)
            lon = [bbox[0], bbox[1]]
            lat = [bbox[2], bbox[3]]
            res = 2000
            xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
            nx = int((xmax - xmin) / res)
            ny = int((ymax - ymin) / res)
            all_location = pd.read_csv('data/FIRMS/fire_nrt_M6_156697.csv')
            fire_data_filter_on_date_and_bbox = all_location[all_location.acq_date.eq(date_of_interest)
                                                             & all_location.latitude.gt(bot_left[0])
                                                             & all_location.latitude.lt(top_right[0])
                                                             & all_location.longitude.gt(bot_left[1])
                                                             & all_location.longitude.lt(top_right[1])]
            timestamp_per_day = fire_data_filter_on_date_and_bbox.acq_time.unique()
            transformer2 = Transformer.from_crs(4326, 32610)
            for l in range(timestamp_per_day.shape[0]):
                # if True:
                #     time_stamp_start, time_stamp_end = convert_int_to_timestamp(timestamp_per_day[l], 3)
                #     img_collection, img_collection_as_gif = dataset_pre.prepare_daily_image(False, satellite='GOES', date_of_interest=date_of_interest, time_stamp_start=time_stamp_start, time_stamp_end=time_stamp_end)
                #     img_to_visualize = img_collection.max()
                #     dataset_pre.download_image_to_gcloud(img_to_visualize.toFloat(), 'GOES', date_of_interest + str(timestamp_per_day[l]))
                fire_data_filter_on_timestamp = np.array(fire_data_filter_on_date_and_bbox[fire_data_filter_on_date_and_bbox.acq_time.eq(timestamp_per_day[l])])

                image_size = (ny, nx)
                #  Create Each Channel
                b1_pixels = np.zeros((image_size), dtype=np.float)
                b2_pixels = np.zeros((image_size), dtype=np.float)
                b3_pixels = np.zeros((image_size), dtype=np.float)
                b4_pixels = np.zeros((image_size), dtype=np.float)

                # coordination in images = (coord_in_crs - min_bbox) / resolution
                for k in range(1, fire_data_filter_on_timestamp.shape[0]):
                    record = fire_data_filter_on_timestamp[k]
                    lon_point = transformer2.transform(record[0], record[1])[0]
                    lat_point = transformer2.transform(record[0], record[1])[1]
                    cord_x = int((lon_point - xmin) / res)
                    cord_y = int((lat_point - ymin) / res)
                    b1_pixels[cord_y, cord_x] = record[2]
                    b2_pixels[cord_y, cord_x] = record[9]
                    b3_pixels[cord_y, cord_x] = record[11]
                    b4_pixels[cord_y, cord_x] = record[12]

                # Geotransform matrix: (top_left_lon, resolution_x, spin, top_left_lat, resolution_y, spin)
                geotransform = (xmin, res, 0, ymin, 0, res)

                # create the n-band raster file
                dst_ds = gdal.GetDriverByName('GTiff').Create('label/' + location + ' label' + '/' + "Cal_fire_" + location + satellite + '-' + str(date_of_interest) + str(timestamp_per_day[l]) + '_downsampled.tif', image_size[1], image_size[0], 4, gdal.GDT_Float64)

                dst_ds.SetGeoTransform(geotransform)  # specify coords
                srs = osr.SpatialReference()  # establish encoding
                srs.ImportFromEPSG(32610)  # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(b1_pixels)  # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(b2_pixels)  # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(b3_pixels)  # write b-band to the raster
                dst_ds.GetRasterBand(4).WriteArray(b4_pixels)  # write b-band to the raster
                dst_ds.FlushCache()  # write to disk
                dst_ds = None