from pprint import pprint

import ee
from utils.EarthEngineMapClient import EarthEngineMapClient
import yaml

ee.Initialize()

class VIIRS:
    def __init__(self):
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')

    def collection_of_interest(self, start_time, end_time, geometry):
        viirs_collection = self.viirs.filterDate(start_time, end_time).filterBounds(geometry)
        vis_params = {'bands': ['M5', 'M4', 'M3'], 'min': 0, 'max': 3000.0}
        return viirs_collection, vis_params