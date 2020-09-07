import ee

ee.Initialize()

class Sentinel1:
    def __init__(self):
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')

    def collection_of_interest(self, start_time, end_time, geometry):
        vh = self.sentinel1\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filterDate(start_time, end_time)\
            .filterBounds(geometry)\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        vhAscending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        vhDescending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

        composite = ee.Image.cat([
            vhAscending.select('VH').mean(),
            ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).mean(),
            vhDescending.select('VH').mean()])\
            .focal_median()

        vis_params = {'bands': ['VH', 'VV', 'VH_1'], 'min': [-25, -20, -25], 'max': [0, 10, 0]}
        return composite, vis_params