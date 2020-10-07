import ee

class GOES:
    def __init__(self):
        self.name = "GOES"
        self.goes_17 = ee.ImageCollection("NOAA/GOES/17/MCMIPF")
        # self.goes_17 = ee.ImageCollection("NOAA/GOES/17/FDCF")
        # self.fire_mask_codes = [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35]
        # self.confidence_values = [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    def collection_of_interest(self, start_time, end_time, geometry):
        goes_17_collection = self.goes_17.filterDate(start_time, end_time).filterBounds(geometry).map(self.applyScaleAndOffset)
        return ee.ImageCollection(goes_17_collection)

    def applyScaleAndOffset(self, image):
        image = ee.Image(image)
        bands = []
        for i in range(1, 6):
            bandName = 'CMI_C' + f'{i:02d}'
            offset = ee.Number(image.get(bandName + '_offset'))
            scale =  ee.Number(image.get(bandName + '_scale'))
            bands.append(image.select(bandName).multiply(scale).add(offset))
            dqfName = 'DQF_C' + f'{i:02d}'
            bands.append(image.select(dqfName))

        for i in range(7, 16):
            bandName = 'CMI_C' + f'{i:02d}'
            offset = ee.Number(image.get(bandName + '_offset'))
            scale =  ee.Number(image.get(bandName + '_scale'))
            bands.append(image.select(bandName).multiply(scale).add(offset).unitScale(250, 400))

            dqfName = 'DQF_C' + f'{i:02d}'
            bands.append(image.select(dqfName))

        return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))

    def get_visualization_parameter(self):
        return {'bands':['CMI_C07', 'CMI_C13', 'CMI_C14'], 'min': 0, 'max': 1.0}

    def apply_resample(self, img):
        return img.resample()