from Preprocessing.PreprocessingService import PreprocessingService
import matplotlib.pyplot as plt

class VisualizationService:
    def __init__(self):
        return

    def scatter_plot(self, location, satellite):
        preprocessing = PreprocessingService()
        dir = 'data/'+satellite + '/'
        arr_palsar, _ = preprocessing.read_tiff(dir+location+'.tif')
        arr_dnbr, _ = preprocessing.read_tiff(dir+location+'_dnbr.tif')
        plt.scatter(arr_dnbr, arr_palsar[0,:,:], c="g", alpha=0.5, linewidths=0.1,
                    label="HV")
        # plt.scatter(arr_dnbr, arr_palsar[1,:,:], c="g", alpha=0.5,
        #             label="HH")
        # plt.scatter(arr_dnbr, arr_palsar[2,:,:], c="g", alpha=0.5,
        #             label="HVHH")
        plt.xlabel("dnbr")
        plt.ylabel("Backscatter gamma")
        plt.legend(loc='upper left')
        plt.show()