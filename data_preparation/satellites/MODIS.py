from pprint import pprint

import ee
from utils.EarthEngineMapClient import EarthEngineMapClient
import yaml

ee.Initialize()

class MODIS:
    def __init__(self):
        self.modis = ee.ImageCollection('MODIS/006/MOD09GQ')

    def collection_of_interest(self, start_time, end_time, geometry):
        modis_collection = self.modis.filterDate(start_time, end_time).filterBounds(geometry)
        vis_params = {'bands': ['sur_refl_b02', 'sur_refl_b02', 'sur_refl_b01'], 'min': -100.0, 'max': 8000.0}
        return modis_collection, vis_params
