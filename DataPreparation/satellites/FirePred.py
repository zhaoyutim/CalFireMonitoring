import datetime

import ee

class FirePred:
    def __init__(self):
        self.name = "FirePred"
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
        self.srtm = ee.Image("CGIAR/SRTM90_V4")
        self.landcover = ee.ImageCollection("MODIS/061/MCD12Q1")
        self.weather = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        self.drought = ee.ImageCollection("GRIDMET/DROUGHT")
        self.viirs = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")
        self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/afall')

    def collection_of_interest(self, start_time, end_time, geometry):
        # Weather Data
        self.weather = self.weather.filterDate(start_time, end_time).filterBounds(geometry)
        precipitation = self.weather.select('pr').median()
        wind_direction = self.weather.select('th').median()
        tmmn = self.weather.select('tmmn').median()
        tmmx = self.weather.select('tmmx').median()
        erc = self.weather.select('erc').median()
        sph = self.weather.select('sph').median()
        wind_velocity = self.weather.select('vs').median()

        # Elevation Data
        elevation = self.srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)

        # Drought Data
        EDDI_14d = self.drought.select('pdsi').filterDate(start_time, end_time).median()
        igbp_land_cover = self.landcover.filterDate(start_time[:4]+'-01-01', start_time[:4]+'-12-31').filterBounds(geometry).select('LC_Type1').median()

        # VIIRS IMG and AF product
        viirs_img = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).select(['M11', 'I2', 'I1']).median()
        viirs_ndvi = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).map(self.get_ratio).select(['ndvi']).median()
        viirs_af_img = self.viirs_af.filterBounds(geometry)\
            .filter(ee.Filter.gte('acq_date', start_time[:-6]))\
            .filter(ee.Filter.lt('acq_date', (datetime.datetime.strptime(end_time[:-6],'%Y-%m-%d') + datetime.timedelta(1)).strftime('%Y-%m-%d'))) \
            .map(self.get_buffer)\
            .reduceToImage(['bright_t31'], ee.Reducer.first())\
            .rename(['af'])
        # .filter(ee.Filter.eq('daynight', 'D'))\
        return ee.ImageCollection(ee.Image([viirs_img, viirs_ndvi, precipitation, wind_velocity, wind_direction, tmmn, tmmx, erc, sph, slope, aspect, elevation, EDDI_14d, igbp_land_cover, viirs_af_img]))

    def get_visualization_parameter(self):
        return {'bands': ['M11', 'I2', 'I1'], 'min': 0, 'max': 6000.0}

    def get_buffer(self, feature):
        return feature.buffer(500/2).bounds()

    def get_ratio(self, image):
        image = ee.Image(image)
        i1 = image.select('I1')
        i2 = image.select('I2')
        ndvi = i2.subtract(i1).divide(i2.add(i1))
        ndvi = ndvi.rename('ndvi')
        return ndvi