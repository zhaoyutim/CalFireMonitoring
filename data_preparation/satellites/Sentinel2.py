import ee

class Sentinel2:
    def __init__(self):
        self.name = "Sentinel2"
        self.sentinel2c = ee.ImageCollection('COPERNICUS/S2')
        self.sentinel2a = ee.ImageCollection('COPERNICUS/S2_SR')
        self.s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        # max_cloud_probability
        self.max_cloud_probability = 100

    def collection_of_interest(self, start_time, end_time, geometry):
        # Here needs to filter boundary on the rectangular or polygons, otherwise we would easily reach the limit(5000) of
        # elements.
        s2a_sr = self.sentinel2a.filterDate(start_time, end_time).filterBounds(geometry).map(self.maskEdges)
        s2_clouds = self.s2_clouds.filterDate(start_time, end_time).filterBounds(geometry)

        s2a_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask') \
            .apply(s2a_sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))
        s2a_cloud_masked_collection = ee.ImageCollection(s2a_sr_with_cloud_mask).map(self.maskClouds)

        return s2a_cloud_masked_collection

    def get_visualization_parameter(self):
        return {'min': 0, 'max': 3000, 'bands': ['B12', 'B11', 'B12']}

    def maskClouds(self, img):
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        isNotCloud = clouds.lt(self.max_cloud_probability)
        return img.updateMask(isNotCloud)

    def maskEdges(self, s2_img):
        return s2_img.updateMask(
            s2_img.select('B8A')
                .mask()
                .updateMask(s2_img.select('B9')
                            .mask())
        )