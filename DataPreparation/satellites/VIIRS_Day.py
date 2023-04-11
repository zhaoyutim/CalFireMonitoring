import ee
import datetime

class VIIRS_Day:
    def __init__(self):
        self.name = "VIIRS_Day"
        # self.viirs = ee.ImageCollection('projects/ee-zhaoyutim/assets/double_creek_fire_day')
        # self.viirs = ee.ImageCollection('projects/ee-zhaoyutim/assets/sydney_fire_day')
        # self.viirs = ee.ImageCollection('projects/grand-drive-285514/assets/swedish_fire')
        self.viirs = ee.ImageCollection('projects/proj5-dataset/assets/proj5_dataset').filter(ee.Filter.stringContains('system:index','IMG'))
        # self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/fire_archive_SV-C2_230093')
        # self.viirs_af = ee.FeatureCollection('users/omegazhangpzh/NRT_AF/SUOMI_VIIRS_C2_Global_Archived_2021')
        # self.viirs_af = ee.FeatureCollection('projects/ee-zhaoyutim/assets/euafall')
        self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/afall')
        # self.viirs_af = ee.FeatureCollection('projects/ee-zhaoyutim/assets/2022naafall09')
        # self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/fire_archive_SV-C2_232183')

        # self.polygon = ee.FeatureCollection("users/zhaoyutim/polygon2019")
        # self.viirs_sr = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
        self.viirs_sr = ee.ImageCollection('projects/proj5-dataset/assets/proj5_dataset').filter(ee.Filter.stringContains('system:index','MOD'))

    def collection_of_interest(self, start_time, end_time, geometry):
        year = start_time[:4]
        self.polygon = ee.FeatureCollection("users/zhaoyutim/polygon"+year).map(self.set_datecurrent)
        viirs_collection = self.viirs.filterDate(start_time, end_time).filterBounds(geometry)
        self.datePolygon = self.polygon.filter(
            ee.Filter.stringContains(leftField='DateCurren', rightValue=start_time[:4] + "/" + start_time[5:7] + "/" + start_time[8:10])).reduceToImage(['OBJECTID'], ee.Reducer.first())
        self.viirs_af_img = self.viirs_af.filterBounds(geometry)\
            .filter(ee.Filter.gte('acq_date', start_time[:-6]))\
            .filter(ee.Filter.lt('acq_date', (datetime.datetime.strptime(end_time[:-6],'%Y-%m-%d')+datetime.timedelta(1)).strftime('%Y-%m-%d'))) \
            .map(self.get_buffer)\
            .reduceToImage(['bright_t31'], ee.Reducer.first())\
            .rename(['af'])
        # .filter(ee.Filter.eq('daynight', 'D'))\
        self.viirs_sr_img = self.viirs_sr.filterDate(start_time, end_time).filterBounds(geometry).select(['b1']).mosaic().rename('m11ImageI')
        # self.viirs_af_img = self.viirs_af.filterBounds(geometry)\
        #     .filter(ee.Filter.gte('ACQ_DATE', ee.Date(start_time[:-6]).millis()))\
        #     .filter(ee.Filter.lt('ACQ_DATE',  ee.Date((datetime.datetime.strptime(end_time[:-6],'%Y-%m-%d')+datetime.timedelta(1)).strftime('%Y-%m-%d')).millis()))\
        #     .filter(ee.Filter.gt('ACQ_TIME', '1200')).map(self.get_buffer)\
        #     .reduceToImage(['BRIGHT_T31'], ee.Reducer.first())\
        #     .rename(['af'])
        return viirs_collection.map(self.get_cloud_masked_img).map(self.add_bands)

    def get_visualization_parameter(self):
        return {'bands': ['b1', 'b2', 'b3'], 'min': 0, 'max': 100.0}

    def get_buffer(self, feature):
        return feature.buffer(375/2).bounds()

    def add_bands(self, img):
        return img.addBands(self.viirs_sr_img).addBands(self.viirs_af_img).addBands(self.datePolygon)

    def get_cloud_masked_img(self, img):
        return img.addBands((img.select(['b1']).gt(30).And(img.select(['b2']).gt(30))).Not().rename(['cloud_mask']))

    def set_datecurrent(self, feature):
        return feature.set({'DateCurren': ee.Date(feature.get('DateCurren')).format('Y/MM/dd')})