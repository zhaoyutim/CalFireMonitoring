import ee

class FIRMS:
    def __init__(self):
        self.name = "Aqua"
        self.firms = ee.ImageCollection('MODIS/006/MOD14A1')

    def collection_of_interest(self, start_time, end_time, geometry):
        firms_collection = self.firms.filterDate(start_time, end_time).filterBounds(geometry)
        return firms_collection.map(self.apply_unitscale)

    def get_visualization_parameter(self):
        return {'min': 0, 'max': 1, 'palette': ['red', 'orange', 'yellow']}

    def apply_resample(self, img):
        return img.resample()

    def apply_unitscale(self, img):
        return img.unitScale(300, 400)