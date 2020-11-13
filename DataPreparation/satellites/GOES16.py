import ee

class GOES16:
    def __init__(self):
        self.name = "GOES16"
        # self.goes_17 = ee.ImageCollection("NOAA/GOES/17/MCMIPF")
        self.goes_16 = ee.ImageCollection("NOAA/GOES/16/MCMIPF")
        # self.fire_mask_codes = [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35]
        # self.confidence_values = [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    def collection_of_interest(self, start_time, end_time, geometry):
        goes_16_collection = self.goes_16.filterDate(start_time, end_time).filterBounds(geometry).map(self.applyScaleAndOffset)
        return ee.ImageCollection(goes_16_collection)

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
            bands.append(image.select(bandName).multiply(scale).add(offset))

            dqfName = 'DQF_C' + f'{i:02d}'
            bands.append(image.select(dqfName))

        return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))

    def get_visualization_parameter(self):
        return {'bands':['index'], 'min': 0, 'max': 500}

    def apply_resample(self, img):
        return img.resample()

    def get_ratio(self, img):
        image = ee.Image(img);
        c07 = image.select('CMI_C07')
        c02 = image.select('CMI_C02')
        c14 = image.select('CMI_C14')
        c15 = image.select('CMI_C15')
        mask = c02.lt(0.02)
        mask2 = c15.lt(250)
        bands = []
        index = c07.subtract(c14).rename('index')
        bands.append(index)
        return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))