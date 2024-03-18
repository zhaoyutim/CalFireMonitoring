file_list_night = glob('data/*/' + satellite_day.replace('VIIRS_Day', 'VIIRS_Night') + '/' + '/2018*.tif')
file_list_night.sort()
file_list_day = glob('data/*/' + satellite_day + '/' + '/2018*.tif')
file_list_day.sort()

file_list_night = glob('data/*/' + satellite_day.replace('VIIRS_Day', 'VIIRS_Night') + '/' + '/2018*.tif')
file_list_night.sort()
file_list_day = glob('data/*/' + satellite_day + '/' + '/2018*.tif')
file_list_day.sort()

for file in file_list_night:
    if file.replace('Night', 'Day') not in file_list_day:
        print(file)