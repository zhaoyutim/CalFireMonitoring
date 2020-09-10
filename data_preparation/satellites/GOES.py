import ee

class GOES:
    def __init__(self, geometry):
        self.geometry = geometry
        self.goes_16 = ee.ImageCollection("NOAA/GOES/16/MCMIPF")
        # self.goes_17 = ee.ImageCollection("NOAA/GOES/17/FDCF")
        # self.fire_mask_codes = [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35]
        # self.confidence_values = [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    def collection_of_interest(self, start_time, end_time, geometry):
        goes_16_data = self.goes_16.filterDate(start_time, end_time).filterBounds(geometry).map(self.applyScaleAndOffset)
        # goes_17_data = self.goes_17.filterDate(start_time, end_time).filterBounds(geometry)

        # goes_16_confidence = goes_16_data.select(['Mask']).map(self.map_from_mask_codes_to_confidence_values)
        # goes_17_confidence = goes_17_data.select(['Mask']).map(self.map_from_mask_codes_to_confidence_values)

        # goes_16_max_confidence = goes_16_confidence.reduce(ee.Reducer.max())
        # goes_17_max_confidence = goes_17_confidence.reduce(ee.Reducer.max())

        # combined = ee.ImageCollection([goes_16_data, goes_17_data])
        return goes_16_data

    # def map_from_mask_codes_to_confidence_values(self, image):
    #     return image.clip(self.geometry).remap(self.fire_mask_codes, self.confidence_values, 0)
    #
    def applyScaleAndOffset(self, image):
        image = ee.Image(image)
        bands = []
        for i in range(1, 16):
            bandName = 'CMI_C' + f'{i:02d}'
            offset = ee.Number(image.get(bandName + '_offset'))
            scale =  ee.Number(image.get(bandName + '_scale'))
            bands.append(image.select(bandName).multiply(scale).add(offset))

            dqfName = 'DQF_C' + f'{i:02d}'
            bands.append(image.select(dqfName))
        green1 = bands[2].multiply(0.45)
        green2 = bands[4].multiply(0.10)
        green3 = bands[0].multiply(0.45)
        green = green1.add(green2).add(green3)
        bands.append(green.rename('GREEN'))

        return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))

    def get_visualization_parameter(self):
        return {'bands':['CMI_C04', 'CMI_C06', 'CMI_C04'], 'min': 0, 'max': 1.3}