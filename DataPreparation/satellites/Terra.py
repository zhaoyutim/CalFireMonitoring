import ee

class FIRMS:
    def __init__(self):
        self.name = "Terra"
        self.firms = ee.ImageCollection('FIRMS')

    def collection_of_interest(self, start_time, end_time, geometry):
        firms_collection = self.firms.filterDate(start_time, end_time).select('T21').filterBounds(geometry)
        return firms_collection.map(self.apply_unitscale)

    def get_visualization_parameter(self):
        return {'min': 0, 'max': 1, 'palette': ['red', 'orange', 'yellow']}

    def apply_resample(self, img):
        return img.resample()

    def apply_unitscale(self, img):
        return img.unitScale(300, 400)