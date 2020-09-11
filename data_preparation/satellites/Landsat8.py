import ee

class Landsat8:
    def __init__(self):
        self.landsat8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')

    def collection_of_interest(self, start_time, end_time, geometry):
        landsat_collection = self.landsat8.filterDate(start_time, end_time).map(self.mask_L8_sr).filterBounds(geometry)
        return landsat_collection

    def mask_L8_sr(self, img):
        cloud_shadow_bit_mask = (1 << 3)
        clouds_bit_mask = (1 << 5)
        qa = img.select('pixel_qa')
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        return img.updateMask(mask)

    def get_visualization_parameter(self):
        return {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}