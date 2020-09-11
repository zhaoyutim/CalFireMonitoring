import ee


class VIIRS:
    def __init__(self):
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')

    def collection_of_interest(self, start_time, end_time, geometry):
        viirs_collection = self.viirs.filterDate(start_time, end_time).filterBounds(geometry)
        return viirs_collection

    def get_visualization_parameter(self):
        return {'bands': ['M11', 'I3', 'M11'], 'min': 0, 'max': 3000.0}