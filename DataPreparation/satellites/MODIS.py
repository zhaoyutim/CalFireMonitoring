import ee

class MODIS:
    def __init__(self):
        self.name = "MODIS"
        self.modis = ee.ImageCollection("MODIS/006/MOD09GA")

    def collection_of_interest(self, start_time, end_time, geometry):
        modis_collection = self.modis.filterDate(start_time, end_time).filterBounds(geometry)
        return modis_collection.map(self.apply_resample)

    def get_visualization_parameter(self):
        return {'bands': ['sur_refl_b07', 'sur_refl_b05', 'sur_refl_b07'], 'min': 0.0, 'max': 3000.0}

    def apply_resample(self, img):
        return img.resample()
