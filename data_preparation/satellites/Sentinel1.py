import datetime

import ee
import yaml

with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class Sentinel1:
    def __init__(self, mode, location):
        self.name = "Sentinel1"
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        self.mode = mode
        self.time_for_master_image = str(config.get(location).get('start') - datetime.timedelta(days=7)) + 'T00:00'
        self.time_for_master_image_end = str(config.get(location).get('start') - datetime.timedelta(days=1)) + 'T00:00'


    def collection_of_interest(self, start_time, end_time, geometry):
        sentinel1_vh_vv = self.sentinel1\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        vh_vv_filtered = sentinel1_vh_vv\
            .filterDate(start_time, end_time)\
            .filterBounds(geometry)\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        vh_vv_ascending = vh_vv_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        vh_vv_descending = vh_vv_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

        if self.mode == "asc":
            self.master_img = sentinel1_vh_vv.filterDate(self.time_for_master_image, self.time_for_master_image_end)\
                .filterBounds(geometry)\
                .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).mean()
            collection = vh_vv_ascending.map(self.get_log_ratio)
        else:
            self.master_img = sentinel1_vh_vv.filterDate(self.time_for_master_image, self.time_for_master_image_end)\
                .filterBounds(geometry)\
                .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).mean()
            collection = vh_vv_descending.map(self.get_log_ratio)

        return collection

    def get_visualization_parameter(self):
        return {'bands': ['VH', 'VV', 'VH'], 'min': [-3, -3, -3], 'max': [0, 0, 0]}

    def get_log_ratio(self, img):
        return ee.Image(self.master_img).subtract(img)