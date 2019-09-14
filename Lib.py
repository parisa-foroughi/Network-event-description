import numpy as np
import re
import plotly.graph_objs as go
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
# from p_tsne import p_tsne as p_tsne_model

def data_loader(fn, scale=False, ft_regex=None):
    """loader for merged.csv"""
    d = np.genfromtxt(fn, dtype=float, delimiter=',', skip_header=1)
    print("Raw data shape (timestamp col incl.): " + str(d.shape))
    tstp = d[:, 0]
    if ft_regex:
        ft_names = get_feature_names_bis(fn)
        ft_filter = re.compile(ft_regex, re.IGNORECASE)
        ft_idx = np.array([i for i, v in enumerate(map(ft_filter.match, ft_names)) if v is not None])
        if len(ft_idx) > 0:
            d = d[:, ft_idx]
        print("Data shape after ft filter (timestamp incl.): " + str(d.shape))
    d = d[:, 1:]
    if scale:
        d = scale_data(d)
    return tstp.tolist(), d

def minmax(d, axis=0):
    """"min-max scaler"""
    range = np.asarray(np.max(d, axis=axis) - np.min(d, axis=axis)).reshape(-1)
    zero_idx = np.argwhere(range == 0)
    range[zero_idx] = 1
    return (d-np.min(d, axis=axis)) / range

def get_feature_names_bis(path, delimiter=','):
    "a more direct and simpler implementation than get_feature_names()"
    with open(path, "r") as f:
        header = f.readline()
    return header.split(delimiter)


def scale_data(d):
    d = d - np.mean(d, axis=0)
    ft_scale = np.std(d, axis=0)
    z_index = np.where(ft_scale < 1e-6)
    ft_scale[z_index] = 1
    d = d / ft_scale
    return d

def data_sanity(data, ft_names, regex_str):
    ft_filter = re.compile(regex_str, re.IGNORECASE)
    ft_names = [ft_names[i] for i, v in enumerate(map(ft_filter.match, ft_names)) if v is not None]
    inval_col = np.where(np.any(np.isnan(data), axis=0))
    data = np.delete(data, inval_col, axis=1)
    ft_names = np.delete(ft_names, inval_col)
    print("Data shape after nan removal: " + str(data.shape))
    inval_col = np.where(np.any(np.isinf(data), axis=0))
    data = np.delete(data, inval_col, axis=1)
    ft_names = np.delete(ft_names, inval_col)
    print("Data shape after inf removal: " + str(data.shape))
    print(len(ft_names), data.shape[1])
    assert len(ft_names) == data.shape[1]
    return data, ft_names

def sorted_ft(pos, data, tstp_rel, win):
    win_idx = [i for i, v in enumerate(tstp_rel) if abs(v - pos) <= win]
    data_slice = data[win_idx, :]
    slice_mean = abs(
        np.mean(data_slice[:int(len(win_idx) / 3), :], axis=0)-np.mean(data_slice[-int(len(win_idx) / 3):, :],axis=0))
    ordered = sorted(enumerate(slice_mean.tolist()), key=lambda s: s[1], reverse=True)
    return ordered

def plot_results(result, tstp, legends, ft_names):
    data_plots = []
    for j in range(len(result)):
        data_plots.append(go.Scatter(
            x=tstp[j],
            y=result[j],
            mode='lines+markers',
            name=legends[j],
            hovertext= ft_names[j]
        ))
    return data_plots

def zoomed_pic(data):
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.plot(data[i][0],data[i][1], data[i][2], label= data[i][3])
    axins = zoomed_inset_axes(ax, 3, loc=1)  # zoom-factor: 2.5, location: upper-left
    # ax1.plot(data[0], data[1], data[2], label=data[3])
    plt.legend('lower left')
    for i in range(len(data)):
        axins.plot(data[i][0],data[i][1], data[i][2])
    x1, x2, y1, y2 = -10, 200, 0.7, 1  # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)  # apply the y-limits
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")


def plot_gen(in1, in2, ft_ordered):
    mark = ['o-', '.-']
    labels = ['enable/disable BFD session', 'enable/disable interface']
    plot_d = []
    for i in range(in1, in2):
        print(i)
        plot_d.append([list(range(len([var for _, var in ft_ordered[i]]))),[var for _, var in ft_ordered[i]], mark[i], labels[i]])
    zoomed_pic(plot_d)
    plt.show()

def gen_plots(data, hovertxt, name):
    fig = go.Figure()
    for i in range(2):
        fig.add_trace(go.Scatter(
            x = data[1][i],
            y = data[0][i],
            mode ='lines+markers',
            hovertext = hovertxt[i],
            name = data[-1][i]
        ))
    fig.update_layout(
        legend=go.layout.Legend(
            x=0.8,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
            bgcolor = 'white',
            bordercolor = "Black",
            borderwidth = 2
        )
    )
    fig.show()