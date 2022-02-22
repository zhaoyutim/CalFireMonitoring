import datetime
import os

import yaml
from easydict import EasyDict
from pyproj import CRS
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, bbox_to_dimensions, CRS, \
    SentinelHubDownloadClient
import matplotlib.pyplot as plt

from LowResSatellitesService.Satellites.MODIS import MODIS
from LowResSatellitesService.Satellites.Sentinel3 import Sentinel3


class LowResSatellitesService:
    def __init__(self):
        with open("LowResSatellitesService/secrets.yaml", "r", encoding="utf8") as f:
            self.secret = yaml.load(f, Loader=yaml.FullLoader)
        with open("config/configuration.yml", "r", encoding="utf8") as f:
            self.fire_locations = yaml.load(f, Loader=yaml.FullLoader)

        self.config = SHConfig()

        if self.config.sh_client_id == '' or self.config.sh_client_secret == '':
            print(
                "Warning! To use Sentinel Hub Process API, please provide the credentials (client ID and client secret).")
            self.registerate_new_id()
        else:
            print('Using Id:{}'.format(self.config.sh_client_id))

    def get_client_from_satellite_name(self, satellites):
        if satellites == "S3":
            return [Sentinel3("TIR"), Sentinel3("SWIR")]
        elif satellites == "MODIS":
            return [MODIS()]
        else:
            raise NameError("No satellite info provided")

    def get_data_collection_from_satellite_name(self, satellites):
        if satellites == "S3":
            return DataCollection.SENTINEL3_SLSTR
        elif satellites == "MODIS":
            return DataCollection.MODIS


    def registerate_new_id(self):
        self.config.instance_id = self.secret.get('sentinel_hub_instance_id')
        self.config.sh_client_id = self.secret.get('sentinel_hub_client_id')
        self.config.sh_client_secret = self.secret.get('sentinel_hub_client_secret')
        self.config.save()

    def fetch_imagery_from_sentinel_hub(self, location, satellites=['S3']):
        start_date = self.fire_locations.get(location)['start']
        end_date = self.fire_locations.get(location)['end']
        timedif = (end_date-start_date).days
        for satellite in satellites:
            if satellite != 'S3':
                continue
            clients = self.get_client_from_satellite_name(satellite)

            for client in clients:
                for time in client.times:
                    for band_name in client.band_names:
                        requests_list = [self.get_request_template(location, satellite, start_date+datetime.timedelta(i), start_date+datetime.timedelta(i), time, band_name, client.units, client.resolution, client.pixel_scale) for i in range(timedif)]
                        requests_list = [request.download_list[0] for request in requests_list]
                        data = SentinelHubDownloadClient(config=self.config).download(requests_list, max_threads=5)
                        for i in range(len(requests_list)):
                            # print((start_date+datetime.timedelta(i)).strftime("%Y%m%d"))
                            tiff_name = requests_list[i].get_storage_paths()[1]
                            os.rename(tiff_name, 'data/'+location +'/' + satellite+'/'+(start_date+datetime.timedelta(i)).strftime("%Y%m%d")+'T'+time+'_'+satellite+'.band'+band_name.replace('B', '')+'.tif')
                            os.remove(requests_list[i].get_storage_paths()[0])
                            os.removedirs(requests_list[i].get_storage_paths()[0].replace('/request.json', ''))

    def fetch_imagery_from_sentinel_hub_with_custom_script(self, location, satellites=['S3']):
        start_date = self.fire_locations.get(location)['start']
        end_date = self.fire_locations.get(location)['end']
        timedif = (end_date-start_date).days
        for satellite in satellites:
            if satellite != 'S3':
                continue
            clients = self.get_client_from_satellite_name(satellite)

            for client in clients:
                for time in client.times:
                    for band_name in client.band_names:
                        requests_list = [self.get_request_template(location, satellite, start_date+datetime.timedelta(i), start_date+datetime.timedelta(i), time, band_name, client.units, client.resolution, client.pixel_scale) for i in range(timedif)]
                        requests_list = [request.download_list[0] for request in requests_list]
                        data = SentinelHubDownloadClient(config=self.config).download(requests_list, max_threads=5)
                        for i in range(len(requests_list)):
                            # print((start_date+datetime.timedelta(i)).strftime("%Y%m%d"))
                            tiff_name = requests_list[i].get_storage_paths()[1]
                            os.rename(tiff_name, 'data/'+location +'/' + satellite+'/'+(start_date+datetime.timedelta(i)).strftime("%Y%m%d")+'T'+time+'_'+satellite+'.band'+band_name.replace('B', '')+'.tif')
                            os.remove(requests_list[i].get_storage_paths()[0])
                            os.removedirs(requests_list[i].get_storage_paths()[0].replace('/request.json', ''))

    def get_request_template(self, location, satellites, start_time, end_time, time, band_name, units, resolution, pixel_scale):
        longitude = self.fire_locations.get(location)['longitude']
        latitude = self.fire_locations.get(location)['latitude']
        rectangular_size = 0.5
        roi = [longitude - rectangular_size, latitude - rectangular_size,
         longitude + rectangular_size, latitude + rectangular_size]
        boundingbox = BBox(bbox=roi, crs=CRS.WGS84)
        bbox_size = bbox_to_dimensions(boundingbox, resolution=resolution)
        evalscript_true_color = """
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: ["{}"],
                        units: "{}"
                    }}],
                    output: {{
                        bands: 1,
                        sampleType: "FLOAT32"
                    }}
                }};
            }}
            function evaluatePixel(sample) {{
                return [{} * sample.{}];
            }}
        """.format(band_name, units, pixel_scale, band_name)
        # if satellites == "S3":
        #     if time == '05':
        #         start_timestamp = 'T00:00:00Z'
        #         end_timestamp = 'T12:59:00Z'
        #     elif time == '17':
        #         start_timestamp = 'T13:00:00Z'
        #         end_timestamp = 'T23:59:00Z'
        #     else:
        #         raise NotImplementedError("Invalid mode")
        # else:
        start_timestamp = 'T00:00:00Z'
        end_timestamp = 'T23:59:00Z'

        request = SentinelHubRequest(
            evalscript=evalscript_true_color,
            data_folder='data/'+location+ '/'+ satellites,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.get_data_collection_from_satellite_name(satellites),
                    time_interval=(start_time.strftime("%Y-%m-%d")+start_timestamp, end_time.strftime("%Y-%m-%d")+end_timestamp)
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=boundingbox,
            size=bbox_size,
            config=self.config
        )
        # print(f'Image shape at {resolution} m resolution: {bbox_size} pixels')
        return request