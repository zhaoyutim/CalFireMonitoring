import ee

class Sentinel1:
    def __init__(self, mode):
        self.name = "Sentinel1"
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        self.mode = mode

    def collection_of_interest(self, start_time, end_time, geometry):
        vh = self.sentinel1\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filterDate(start_time, end_time)\
            .filterBounds(geometry)\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        vh_ascending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        vh_descending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

        if self.mode == "asc":
            collection = vh_ascending
        else:
            collection = vh_descending

        return collection

    def get_visualization_parameter(self):
        return {'bands': ['VH', 'VV', 'VH'], 'min': [-25, -20, -25], 'max': [0, 10, 0]}