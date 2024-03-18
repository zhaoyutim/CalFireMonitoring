import ee
import datetime

class VIIRS_Night:
    def __init__(self):
        self.name = "VIIRS_Night"
    def collection_of_interest(self, start_time, end_time, geometry):
        self.viirs = ee.ImageCollection('projects/proj5-dataset/assets/proj5_dataset_night').filter(
            ee.Filter.stringContains('system:index', 'IMG')).filterDate(start_time, end_time).filterBounds(geometry)
        return ee.ImageCollection(self.viirs)

    def get_visualization_parameter(self):
        return {'bands': ['b1', 'b2', 'b3'], 'min': 0, 'max': 100.0}
