import datetime
import os
from glob import glob
from pathlib import Path

import rasterio as rasterio
import yaml
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
import pandas as pd
import numpy as np

class FirmsProcessor:
    def __init__(self):
        directory = "data/FIRMS/fire_archive_mid-2018-mid-2020.csv"
        self.fire_pixels = pd.read_csv(directory)
        with open("config/configuration.yml", "r", encoding="utf8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read()
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))

    def firms_generation_from_csv_to_tiff(self, start_time, end_time, location, crs=32610):
        time_dif = end_time - start_time
        latitude = self.config.get(location).get('latitude')
        longitude = self.config.get(location).get('longitude')
        rectangular_size = self.config.get('rectangular_size')
        res = 375
        directory = "data/FIRMS/" + location
        if not os.path.exists(directory):
            os.mkdir(directory)
        transformer = Transformer.from_crs(4326, crs)
        bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
        top_right = [latitude + rectangular_size, longitude + rectangular_size]
        bottom_left_utm = [int(transformer.transform(bottom_left[0], bottom_left[1])[0]), int(transformer.transform(bottom_left[0], bottom_left[1])[1])]
        top_right_utm = [int(transformer.transform(top_right[0], top_right[1])[0]), int(transformer.transform(top_right[0], top_right[1])[1])]
        top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % 375, top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % 375]
        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))

            lon = [bottom_left_utm[0], top_right_utm[0]]
            lat = [bottom_left_utm[1], top_right_utm[1]]
            xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
            nx = int((xmax - xmin) // res)
            ny = int((ymax - ymin) // res)

            fire_data_filter_on_date_and_bbox =  self.fire_pixels[ self.fire_pixels.acq_date.eq(date_of_interest)
                                                             &  self.fire_pixels.latitude.gt(bottom_left[0])
                                                             &  self.fire_pixels.latitude.lt(top_right[0])
                                                             &  self.fire_pixels.longitude.gt(bottom_left[1])
                                                             &  self.fire_pixels.longitude.lt(top_right[1])]
            transformer2 = Transformer.from_crs(4326, crs)

            # for time in range(len(fire_data_filter_on_date_and_bbox.acq_time.unique())):
            #     timestamp_per_day = fire_data_filter_on_date_and_bbox.acq_time.unique()[time]
            for date in range(len(fire_data_filter_on_date_and_bbox.acq_date.unique())):
                fire_date = fire_data_filter_on_date_and_bbox.acq_date.unique()[date]
                fire_data_filter_on_timestamp = np.array(
                    fire_data_filter_on_date_and_bbox[fire_data_filter_on_date_and_bbox.acq_date.eq(fire_date)])
                image_size = (ny, nx)
                #  Create Each Channel
                b1_pixels = np.zeros((image_size), dtype=np.float)
                b2_pixels = np.zeros((image_size), dtype=np.float)
                b3_pixels = np.zeros((image_size), dtype=np.float)
                b4_pixels = np.zeros((image_size), dtype=np.float)

                for k in range(1, fire_data_filter_on_timestamp.shape[0]):
                    record = fire_data_filter_on_timestamp[k]
                    lon_point = transformer2.transform(record[0], record[1])[0]
                    lat_point = transformer2.transform(record[0], record[1])[1]
                    cord_x = int((lon_point - xmin) // res)
                    cord_y = int((lat_point - ymin) // res)
                    if cord_x >= nx or cord_y >= ny:
                        continue
                    b1_pixels[cord_y, cord_x] = max(b1_pixels[cord_y, cord_x], record[2])
                    if record[8] == 'n':
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    elif record[8] == 'h':
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    else:
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    b3_pixels[cord_y, cord_x] = max(b3_pixels[cord_y, cord_x], record[11])
                    b4_pixels[cord_y, cord_x] = max(b4_pixels[cord_y, cord_x], record[12])

                geotransform = (xmin, res, 0, ymin, 0, res)

                dst_ds = gdal.GetDriverByName('GTiff').Create(
                    'data/FIRMS/' + location + '/' + location + 'FIRMS' + '-' + str(
                        date_of_interest)+ '.tif', image_size[1],
                    image_size[0], 4,
                    gdal.GDT_Float32)
                dst_ds.SetGeoTransform(geotransform)  # specify coords
                srs = osr.SpatialReference()  # establish encoding
                srs.ImportFromEPSG(crs)  # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(b1_pixels)  # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(b2_pixels)  # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(b3_pixels)  # write b-band to the raster
                dst_ds.GetRasterBand(4).WriteArray(b4_pixels)  # write b-band to the raster
                dst_ds.FlushCache()  # write to disk
                dst_ds = None

    def accumulation(self, location):
        directory = "data/FIRMS/" + location
        directory_progression = "data/progression/" + location
        if not os.path.exists(directory_progression):
            os.mkdir(directory_progression)
        data_path = Path(directory)
        data_file_list = glob(str(data_path / "*.tif"))
        data_file_list.sort()
        template, _ = self.read_tiff(data_file_list[0])
        acc_arr = np.zeros(template.shape)
        for i in range(len(data_file_list)):
            current_time = data_file_list[i][-14:-4]
            tiff_arr, profile = self.read_tiff(data_file_list[i])
            acc_arr = tiff_arr + acc_arr
            profile.data['count'] = 1
            self.write_tiff(directory_progression+'/'+location+current_time+'.tif', acc_arr[np.newaxis, 2,:,:]>0, profile)


