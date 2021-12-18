from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
# sizes: xx-small, x-small, small, medium, large, x-large, xx-large

size = 'xx-large'
params = {
    'font.family': 'Inter',
    'legend.fontsize': size,
    'axes.labelsize': size,
    'axes.titlesize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
}
plt.rcParams.update(params)


def plot(name, xs, ys):
    plt.plot(xs, ys)
    plt.xlabel('Time (hours)')
    plt.ylabel('# Instances')
    plt.xticks(rotation='25')
    # plt.xticks(range(0, duration//self.hour + 1, 12))
    # plt.hlines(result.average_instances, 0, duration //
    # 		self.hour, color='tab:blue', linestyles='dashed')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.axis([0, duration//self.hour, 0, 64])

    plt.savefig(
        'gcp-{}.pdf'.format(name),
        bbox_inches='tight',
        pad_inches=0
    )
    # plt.show()
    plt.clf()


def parse_trace_log(filename):
    xs, ys = [], []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            ts = line[:20]
            ts = datetime.strptime(ts, '%Y-%m-%d, %H:%M:%S')
            nr_instances = int(line.split(',')[-1])
            xs.append(ts)
            ys.append(nr_instances)

    return xs, ys


def plot_fig1():
    name = 'v100x1-east1c'
    xs, ys = parse_trace_log(name + '.txt')
    plot(name, xs, ys)


def plot_fig2():
    name = 'a100x1-east1b'
    xs, ys = parse_trace_log(name + '.txt')
    plot(name, xs, ys)


plot_fig1()
plot_fig2()