import datetime

import ee
import yaml

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class Sentinel1:
    def __init__(self, mode, location):
        self.name = "Sentinel1"
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
        self.mode = mode
        self.time_for_master_image = str(config.get(location).get('start') - datetime.timedelta(days=15)) + 'T00:00'
        self.time_for_master_image_end = str(config.get(location).get('start') - datetime.timedelta(days=1)) + 'T00:00'


    def collection_of_interest(self, start_time, end_time, geometry):
        self.geometry = geometry
        sentinel1_vh_vv = self.sentinel1\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        vh_vv_filtered = sentinel1_vh_vv\
            .filterBounds(geometry) \
            .filter(ee.Filter.eq('relativeOrbitNumber_start', 168))
        Pre_Imgs = vh_vv_filtered.filterDate(ee.Date(start_time).advance(-12, "month"), ee.Date(start_time)).map(self.imgConv)
        Post_Imgs = vh_vv_filtered.filterDate(ee.Date(start_time), ee.Date(end_time)).map(self.imgConv)
        All_Imgs = vh_vv_filtered.filterDate(ee.Date(start_time).advance(-12, "month"),ee.Date(end_time)).map(self.imgConv)
        self.pre_median_img = Pre_Imgs.median().clip(geometry)
        self.pre_stddev_img = Pre_Imgs.reduce(ee.Reducer.stdDev()).clip(geometry).rename(["VV","VH","angle"])
        kmap_bin_col = ee.ImageCollection(Post_Imgs.map(self.get_kmap_bin).map(self.get_ratio).map(self.morphological).copyProperties(Post_Imgs, Post_Imgs.propertyNames()))
        # vh_vv_ascending = vh_vv_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        # vh_vv_descending = vh_vv_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        #
        # if self.mode == "asc":
        #     self.master_img = sentinel1_vh_vv.filterDate(self.time_for_master_image, self.time_for_master_image_end)\
        #         .filterBounds(geometry)\
        #         .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        #         .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).mean().unitScale(-25, 5)
        #     collection = vh_vv_ascending.map(self.get_unit_scale).map(self.get_log_ratio)
        # else:
        #     self.master_img = sentinel1_vh_vv.filterDate(self.time_for_master_image, self.time_for_master_image_end)\
        #         .filterBounds(geometry)\
        #         .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        #         .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).mean().unitScale(-25, 5)
        #     collection = vh_vv_descending.map(self.get_unit_scale).map(self.get_log_ratio)

        return kmap_bin_col.select('VV')

    def get_visualization_parameter(self):
        return {'bands': ['VV', 'VV', 'VV'], 'min': 0, 'max': 1}

    def get_log_ratio(self, img):
        return ee.Image(self.master_img).subtract(img)

    def get_unit_scale(self, img):
        return img.unitScale(-25, 5)

    def imgConv(self, img):
        G_kernel = ee.Kernel.gaussian(11)
        return img.log10().multiply(10.0).convolve(G_kernel).copyProperties(img, img.propertyNames())
    def get_kmap_bin(self, img):
      return ee.Image(img.subtract(self.pre_median_img).divide(self.pre_stddev_img).copyProperties(img, img.propertyNames()))

    def get_ratio(self, img):
        mean_unburn = img.reduceRegion(reducer=ee.Reducer.mean(),
                                       geometry=self.geometry,
                                       maxPixels=1e9,
                                       scale=20).toImage()
        return ee.Image(img.subtract(mean_unburn).clamp(0, 1).copyProperties(img, img.propertyNames()))
    def morphological(self, img):
        kernel = ee.Kernel.circle(radius=1)
        return ee.Image(img.focal_min(kernel=kernel, iterations=2)
                        .focal_max(kernel=kernel, iterations=2)
                        .reproject(crs=img.projection().crs(), scale=20)
                        .copyProperties(img, img.propertyNames()))