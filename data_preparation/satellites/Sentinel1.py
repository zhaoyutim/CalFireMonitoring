import ee

ee.Initialize()

class Sentinel1:
    def __init__(self):
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')

    def collection_of_interest(self, start_time, end_time, geometry):
        sentinel1_collection = self.sentinel1\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .select('VV')\
            .filterDate(self.start_time, self.end_time)\
            .filterBounds(self.geometry)\
            .map(self.mask_edge)

        vis_params = {min: -25, max: 5}
        return sentinel1_collection, vis_params

    def mask_edge(self, img):
        edge = img.lt(-30.0)
        masked_image = img.mask().And(edge.Not())
        return img.updateMask(masked_image)