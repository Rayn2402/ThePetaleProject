
import math
from matplotlib import pyplot as plt
from os import makedirs
from os.path import join
from settings.paths import Paths


def drawBarhChart(data_y, data_x, label_y, label_x, title, filename, foldername):
    """
    Function that generates a barh chart using matplot lib

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
    fig = plt.figure(figsize=(15, 12.5))

    # specifying the type of the chart and the data in it
    plt.barh(data_y, data_x, color="#874bf2")

    # specifying the labels of the chart
    plt.ylabel(label_y)
    plt.xlabel(label_x)

    # specifying the title of the chart
    plt.title(title, fontsize=15)

    # saving the chart in a file in the folder missing_data_charts
    directory = join(Paths.DESC_CHARTS.value, foldername)
    makedirs(directory, exist_ok=True)
    plt.savefig(join(directory, f"{filename}.png"))
    plt.close()


def drawBinaryGroupedBarChart(data_x, group_1, group_2, label_x, label_y, title, filename, foldername):
    """
    Function that generates a grouped bar chart using matplotlib

        :param data_x: the data to be in axe x
        :param group_1: the data of the first group
        :param group_2:  the data of the second group
        :param label_x:  label on the x axis
        :param label_y:  label on the y axis
        :param title:  title of the chart
        :param filename:  the file where the figure containing the chart will be saved
        :param foldername:  the folder where the figure containing the chart will be saved

        :generate a figure containing the chart from the data given
    """
    # width of the bar
    w = 0.4

    # we plot two bars, one for male and one for females
    plt.bar(data_x, group_1["values"], w,
            label=group_1["label"], color="#874bf2")
    plt.bar(data_x, group_2["values"], w,
            bottom=group_1["values"], label=group_2["label"], color="#55f1a0")

    # specifying the labels of the chart
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    # specifying the title of the chart
    plt.title(title)

    # showing the legend
    plt.legend()

    # saving the chart in a file in the folder missing_data_charts
    directory = join(Paths.DESC_CHARTS.value, foldername)
    makedirs(directory, exist_ok=True)
    plt.savefig(join(directory, f"{filename}.png"))
    plt.close()


def drawHistogram(data, label_x, title, filename, foldername):
    """
    Function that generates a histogram charts using matplotlin

        :param data: the data to plot
        :param label_x:  label on the axe x
        :param title:  title of the chart
        :param filename:  the file where the figure containing the chart will be saved
        :param foldername:  the folder where the figure containing the chart will be saved

        :generate a figure containing the chart from the data given
    """
    num_bins = math.ceil(math.sqrt(len(data[label_x].values)))
    plt.hist(data[label_x].dropna(), num_bins,
             facecolor='#63ffbb', density=True, edgecolor='white', linewidth=1.2)

    # specifying the title of the chart
    plt.title(title, fontsize=15)

    # saving the chart in a file in teh folder missing_data_charts
    directory = join(Paths.DESC_CHARTS.value, foldername)
    makedirs(directory, exist_ok=True)
    plt.savefig(join(directory, f"{filename}.png"))
    plt.close()