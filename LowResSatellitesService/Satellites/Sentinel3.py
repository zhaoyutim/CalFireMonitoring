class Sentinel3:
    def __init__(self, mode):
        if mode == "TIR":
            self.resolution = 1000
            self.times = ["05", "17"]
            self.band_names = ["S7", "S8", "F1", "F2"]
            self.units = "DN"
            self.pixel_scale = 10
        elif mode == "SWIR":
            self.resolution = 500
            self.times = ["17"]
            self.band_names = ["S6"]
            self.units = "REFLECTANCE"
            self.pixel_scale = 10000
