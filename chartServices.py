from matplotlib import pyplot as plt
import os


def drawBarhChart(data_y,data_x,label_y,label_x,title,filename,foldername):
    """
    Function that generate a barh chart using matplot lib

        :param data_y: the data to be in axe y 
        :param data_x: the data to be in axe x
        :param label_y:  label on the axe y
        :param label_x:  label on the axe x
        :param title:  title of the chart
        :param filename:  the file where the figure containing the chart will be saved
        :param foldername:  the folder where the figure containing the chart will be saved

        :generate a figure containing the chart from the data given
    """
    # specifying the figure size
    fig = plt.figure(figsize=(40, .7*len(data_x)))

    # specifying the type of the chart and the data in it
    plt.barh(data_y, data_x, color="#874bf2")

    # specifying the labels of the chart
    plt.ylabel(label_y)
    plt.xlabel(label_y)

    # specifying the title of the chart
    plt.title(title,fontsize=15)

    # saving the chart in a file in teh folder missing_data_charts
    if not os.path.exists(f'{foldername}'):
        os.makedirs(f'{foldername}')
    plt.savefig(
        f'{foldername}/{filename}.png')
    plt.close(fig)