import ee

class MODIS:
    def __init__(self):
        self.modis = ee.ImageCollection("MODIS/006/MOD09GA")

    def collection_of_interest(self, start_time, end_time, geometry):
        modis_collection = self.modis.filterDate(start_time, end_time).filterBounds(geometry)
        return modis_collection

    def get_visualization_parameter(self):
        return {'bands': ['sur_refl_b07', 'sur_refl_b05', 'sur_refl_b07'], 'min': -100.0, 'max': 8000.0}
