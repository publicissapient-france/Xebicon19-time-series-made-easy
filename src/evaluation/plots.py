import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import os

import src.constants.columns as c
import src.constants.files as files


def plot_consumptions(region_df_dict, year, month):
    matplotlib.rcParams.update({'font.size': 22})

    plt.figure(1, figsize=(25, 12))
    for region in region_df_dict.keys():
        df_region = region_df_dict[region]
        df_region[c.EnergyConso.DATE_HEURE] = df_region.index
        df_region = df_region[(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.year)==year)
                           &(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.month==month))]
        plt.plot(df_region[c.EnergyConso.DATE_HEURE], df_region[c.EnergyConso.CONSUMPTION], label=region)
        plt.ylabel(c.EnergyConso.CONSUMPTION)

    plt.legend()
    plt.savefig(os.path.join(files.PLOTS, "Consos énergie France {}-{}.png".format(year, month)))

    plt.close()
