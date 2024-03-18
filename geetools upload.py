import os

import ee
ee.Initialize()
from geetools import batch
if __name__=='__main__':
    img = ee.ImageCollection("projects/proj5-dataset/assets/proj5_dataset").filter(
        ee.Filter.stringContains('system:index', 'IMG'))
    mod = ee.ImageCollection("projects/proj5-dataset/assets/proj5_dataset").filter(
        ee.Filter.stringContains('system:index', 'MOD'))
    img_ids = img.aggregate_array('system:id').getInfo()
    mod_ids = mod.aggregate_array('system:id').getInfo()
    for i in range(len(img_ids)):
        if img_ids[i].replace('IMG', 'MOD') not in mod_ids:
            print(img_ids[i])
    for i in range(len(mod_ids)):
        if mod_ids[i].replace('MOD', 'IMG') not in img_ids:
            print(mod_ids[i])
    print('over')