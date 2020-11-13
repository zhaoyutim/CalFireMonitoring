import ee

class GOES_FIRE:
    def __init__(self):
        self.goes_16 = ee.ImageCollection("NOAA/GOES/16/FDCF")
        self.goes_17 = ee.ImageCollection("NOAA/GOES/17/FDCF")
        self.fire_mask_codes = [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35]
        self.confidence_values = [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    def get_visualization_parameter(self):
        return {'palette': ['white', 'yellow', 'orange', 'red', 'purple'], 'min': 0, 'max': 1}

    def collection_of_interest(self, start_time, end_time, geometry):
        self.geometry = geometry
        goes_16_data = self.goes_16.filterDate(start_time, end_time).filterBounds(geometry)
        goes_17_data = self.goes_17.filterDate(start_time, end_time).filterBounds(geometry)

        goes_16_confidence = goes_16_data.select(['Mask']).map(self.map_from_mask_codes_to_confidence_values)
        goes_17_confidence = goes_17_data.select(['Mask']).map(self.map_from_mask_codes_to_confidence_values)

        goes_16_max_confidence = goes_16_confidence.reduce(ee.Reducer.max())
        goes_17_max_confidence = goes_17_confidence.reduce(ee.Reducer.max())

        combined_confidence = ee.ImageCollection([goes_16_max_confidence, goes_17_max_confidence])

        return combined_confidence

    def map_from_mask_codes_to_confidence_values(self, image):
        return image.clip(self.geometry).remap(self.fire_mask_codes, self.confidence_values, 0)