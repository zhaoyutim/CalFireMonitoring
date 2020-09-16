import datetime
import os
import cv2
import imageio
import yaml

with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    satellites = ['GOES']
    location = 'creek_fire'
    locations = ['North_complex_fire']
    for location in locations:
        start_time = config.get(location).get('start')
        # end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        filenames = []
        time_dif = end_time - start_time

        for i in range(1):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                if os.path.isfile('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    # 加载背景图片
                    bk_img = cv2.imread('images_for_gif/' + location + '/' + satellite + str(date_of_interest) + '.jpg')
                    # 在图片上添加文字信息
                    cv2.putText(bk_img, location + '-' + satellite + '-' + str(date_of_interest), (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    # 显示图片
                    cv2.imshow("add_text", bk_img)
                    cv2.waitKey()
                    # # 保存图片
                    cv2.imwrite("add_text.jpg", bk_img)
                    filenames.append(satellite + str(date_of_interest))