import folium
import ee
import webbrowser


class eeMapClient:
    def __init__(self, location):
        # Add EE drawing method to folium.
        folium.Map.add_ee_layer = self.add_ee_layer
        self.latitude = location.get('latitude')
        self.longitude = location.get('longitude')
        self.map = folium.Map(location=[self.latitude, self.longitude], zoom_start=12)

    def add_ee_layer(self, ee_image_object, vis_params, name):
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
