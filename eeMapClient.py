from pprint import pprint

import folium
import ee
import webbrowser
import tensorflow as tf
import time
import yaml

# Load configuration file
with open("configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class eeMapClient:
    def __init__(self, location):
        # Add EE drawing method to folium.
        folium.Map.add_ee_layer = self.add_ee_layer
        self.latitude = location.get('latitude')
        self.longitude = location.get('longitude')
        self.map = folium.Map(location=[self.latitude, self.longitude], zoom_start=12)

    def add_ee_layer(self, ee_image_object, vis_params, name):
        '''
        :param ee_image_object: Image in GEE
        :param vis_params: Visualisation parameters
        :param name: The name of the Layer, as it will appear in LayerControls
        :return:
        '''
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr="Map Data © Google Earth Engine",
            name=name,
            overlay=True,
            control=True
        ).add_to(self.map)

    def add_to_map(self, img, vis_params, name):
        self.map.add_ee_layer(img, vis_params, name)
        self.map.add_child(folium.LayerControl())
        outHtml = '/Users/zhaoyu/PycharmProjects/CalFireMonitoring/map.html'
        self.map.save(outHtml)
        webbrowser.open('file://' + outHtml)

    def download_image_to_gcloud(self, img, size, filename_prefix):
        '''
        Export images to google cloud, the output image is a rectangular with the center at given latitude and longitude
        :param img: Image in GEE
        :param size: The longth beteen the edge of the rectangular and the center
        :param filename_prefix: The filename prefix in Google cloud
        :return: None
        '''
        export_region = ee.Geometry.Rectangle([self.longitude - size,
                                               self.latitude - size,
                                               self.longitude + size,
                                               self.latitude + size])

        pprint({'Image info:': img.getInfo()})
        print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + config.get('output_bucket'))
              else 'Can not find output Cloud Storage bucket.')

        # Setup the task.
        image_task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description='Image Export',
            fileNamePrefix=filename_prefix,
            bucket=config.get('output_bucket'),
            scale=30,
            fileFormat='GeoTIFF',
            region=export_region.toGeoJSON()['coordinates'],
        )

        image_task.start()

        while image_task.active():
            print('Polling for task (id: {}).'.format(image_task.id))
            time.sleep(30)
        print('Done with image export.')

    def download_image_collection_to_gcloud(self, img_collection, size, filename_prefix, feature_names):
        '''
        Export images to google cloud, the output image is a rectangular with the center at given latitude and longitude
        :param img: Image in GEE
        :param size: The longth beteen the edge of the rectangular and the center
        :param filename_prefix: The filename prefix in Google cloud
        :return: None
        '''
        export_region = ee.Geometry.Rectangle([self.longitude - size,
                                               self.latitude - size,
                                               self.longitude + size,
                                               self.latitude + size])

        pprint({'Image info:': img_collection.getInfo()})
        print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + config.get('output_bucket'))
              else 'Can not find output Cloud Storage bucket.')

        # Setup the task.
        collection_task = ee.batch.Export.table.toCloudStorage(
            collection=img_collection,
            description='Training Export',
            fileNamePrefix=filename_prefix,
            bucket=config.get('output_bucket'),
            fileFormat='TFRecord',
            selectors=feature_names
        )

        collection_task.start()

        while collection_task.active():
            print('Polling for task (id: {}).'.format(collection_task.id))
            time.sleep(30)
        print('Done with image export.')
