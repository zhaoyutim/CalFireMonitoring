import datetime

import ee
import ee.mapclient
import yaml

ee.Initialize()

if __name__ == '__main__':
    with open("config/configuration.yml", "r", encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    time = config.get('LNU_lighting_complex').get('start')
    time2 = config.get('LNU_lighting_complex').get('end')
    timedif = time2-time
    print(str(timedif.days))

    print(time + datetime.timedelta(days=1))