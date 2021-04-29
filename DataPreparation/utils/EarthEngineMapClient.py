import webbrowser

import ee
import folium
import yaml

# Load configuration file
with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class EarthEngineMapClient:
    def __init__(self, location):
        # Add EE drawing method to folium.
        self.location = location
        folium.Map.add_ee_layer = self.add_ee_layer
        self.latitude = config.get(self.location).get('latitude')
        self.longitude = config.get(self.location).get('longitude')
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

    def initialize_map(self):
        self.map.add_child(folium.LayerControl())
        outHtml = "/Users/zhaoyu/PycharmProjects/CalFireMonitoring/map.html"
        self.map.save(outHtml)
        webbrowser.open('file://' + outHtml)
