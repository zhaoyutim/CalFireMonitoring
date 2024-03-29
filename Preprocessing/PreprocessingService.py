import numpy as np
import rasterio
from rasterio.merge import merge
import yaml

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class PreprocessingService:

    def padding(self, coarse_arr, array_to_be_downsampled):
        array_to_be_downsampled = np.pad(array_to_be_downsampled, ((0, 0), (0, coarse_arr.shape[1] * 2 - array_to_be_downsampled.shape[1]), (0, coarse_arr.shape[2] * 2 - array_to_be_downsampled.shape[2])), 'constant', constant_values = (0, 0))
        return array_to_be_downsampled

    def down_sampling(self, input_arr):
        return np.mean(input_arr)

    def standardization(self, array):
        n_channels = array.shape[0]
        for i in range(n_channels):
            nanmean = np.nanmean(array[i, :, :])
            array[i, :, :] = np.nan_to_num(array[i, :, :], nan=nanmean)
            array[i,:,:] = (array[i,:,:]-array[i,:,:].mean())/array[i,:,:].std()
        return np.nan_to_num(array)

    def normalization(self, array):
        n_channels = array.shape[0]
        for i in range(n_channels):
            array[i,:,:] = (array[i,:,:]-np.nanmin(array[i,:,:]))/(np.nanmax(array[i,:,:])-np.nanmin(array[i,:,:]))
        return np.nan_to_num(array)

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read()
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))

    def mosaic_geotiffs(self, geotiff_files):
        # Read images and metadata
        src_files = [rasterio.open(file) for file in geotiff_files]

        # Merge images using maximum values for overlapping locations
        mosaic, out_transform = merge(src_files, method="max")

        # Copy metadata from the first file
        out_meta = src_files[0].meta.copy()

        # Update metadata with the mosaic dimensions and transform
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })

        # Close source files
        for src in src_files:
            src.close()

        return mosaic, out_meta

if __name__=='__main__':
    preprocessing_service = PreprocessingService()
    output_path = 'mosaic.tif'
    tiff_files = ['../data/22713339/VIIRS_Day/2019-03-31_VIIRS_Day.tif', '../data/22938749/VIIRS_Day/2019-06-08_VIIRS_Day.tif']
    mosaic, mosaic_metadata = preprocessing_service.mosaic_geotiffs(tiff_files)
    preprocessing_service.write_tiff(output_path, mosaic, mosaic_metadata)