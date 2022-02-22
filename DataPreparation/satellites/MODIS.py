import ee

class MODIS:
    def __init__(self):
        self.name = "MODIS"
        self.modis = ee.ImageCollection("MODIS/006/MOD09GA")

    def collection_of_interest(self, start_time, end_time, geometry):
        modis_collection = self.modis.filterDate(start_time, end_time).filterBounds(geometry)
        return modis_collection.map(self.apply_resample).map(self.get_bands)

    def get_visualization_parameter(self):
        return {'bands': ['sur_refl_b07', 'sur_refl_b02', 'sur_refl_b01'], 'min': 0.0, 'max': 3000.0}

    def apply_resample(self, img):
        return img.resample()

    def get_bands(self, img):
        b1 = img.select('sur_refl_b01')
        b2 = img.select('sur_refl_b02')
        b7 = img.select('sur_refl_b07')
        return ee.Image.cat([b7, b2, b1])
