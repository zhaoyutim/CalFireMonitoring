import datetime
import os
import cv2
import imageio
import yaml

with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    satellites = ['GOES', 'MODIS', 'VIIRS', 'Sentinel2']
    locations = ['LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex',
                 'August_complex', 'Beachie_wildfire', 'Beachie_wildfire_2', 'Holiday_farm_wildfire',
                 'North_complex_fire', 'North_complex_fire_2', 'North_complex_fire_3',
                 'Cold_spring_fire', 'Doctor_creek_fire']
    for location in locations:
        start_time = config.get(location).get('start')
        # end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        filenames = []
        time_dif = end_time - start_time

        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                if os.path.isfile('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    bk_img = cv2.imread('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg')
                    cv2.putText(bk_img, location + '-' + satellite + '-' + str(date_of_interest), (150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite('images_after_processing/' + location + '/' + satellite + str(date_of_interest) +'.jpg', bk_img)
                    filenames.append(satellite + str(date_of_interest))
        images = []
        for filename in filenames:
            images.append(imageio.imread('images_after_processing/'+ location + '/' + filename + '.jpg'))
        imageio.mimsave('images_after_processing/gif/'  + '/' + location + '.gif', images, format='GIF', fps=1)