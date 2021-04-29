import ee


class VIIRS:
    def __init__(self):
        self.name = "VIIRS"
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')

    def collection_of_interest(self, start_time, end_time, geometry):
        viirs_collection = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).select(['M11', 'I2', 'I1'])
        return viirs_collection

    def get_visualization_parameter(self):
        return {'bands': ['M11', 'I2', 'I1'], 'min': 0, 'max': 6000.0}