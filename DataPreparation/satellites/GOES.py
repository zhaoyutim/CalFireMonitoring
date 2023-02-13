import ee

class GOES:
    def __init__(self):
        self.name = "GOES"
        self.goes_17 = ee.ImageCollection("NOAA/GOES/17/MCMIPF")
        self.goes_16 = ee.ImageCollection("NOAA/GOES/16/MCMIPF")
        # self.fire_mask_codes = [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35]
        # self.confidence_values = [1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    def collection_of_interest(self, start_time, end_time, geometry):
        goes_17_collection = self.goes_17.filterDate(start_time, end_time).filterBounds(geometry).map(self.applyScaleAndOffset).select(['CMI_C07','CMI_C14','CMI_C15'])#.map(self.get_ratio)
        goes_16_collection = self.goes_16.filterDate(start_time, end_time).filterBounds(geometry).map(self.applyScaleAndOffset).select(['CMI_C07','CMI_C14','CMI_C15'])#.map(self.get_ratio)#.map(self.get_ratio)
        # goes_16_max_confidence = goes_16_collection.reduce(ee.Reducer.max())
        # goes_17_max_confidence = goes_17_collection.reduce(ee.Reducer.max())
        #
        # combined_confidence = ee.ImageCollection([goes_16_max_confidence, goes_17_max_confidence])

        # return ee.ImageCollection([combined_confidence.min()])
        return goes_17_collection

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
        return {'bands':['CMI_C07', 'CMI_C14', 'CMI_C15'], 'min': 280, 'max': 400}

    def apply_resample(self, img):
        return img.resample()

    def get_ratio(self, img):
        image = ee.Image(img)
        c07 = image.select('CMI_C07')
        c14 = image.select('CMI_C14')
        c15 = image.select('CMI_C15')
        mask = c15.gt(280)
        index = c07.subtract(c14).divide(c07.add(c14)).multiply(100).updateMask(mask).rename('index')
        return index