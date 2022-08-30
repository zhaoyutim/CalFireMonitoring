class MODIS:
    def __init__(self):
        self.resolution = 500
        self.times = ["1730"]
        self.band_names = ["B07", "B06"]
        self.units = "REFLECTANCE"
        self.pixel_scale = 10000
