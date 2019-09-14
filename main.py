from Lib import *

from plotly.offline import plot


fn = "preprocessed_offline_190519-700-leaf7.csv"
ef = "events_190519_700.csv"
regex_str1 = "^(?!.*(process-info|id|address|port)).*"
regex_str2 = "^(?!.*(process-info|id|address|port|tcp|hundred)).*"
tstp1, data1 = data_loader(fn, scale=False, ft_regex=regex_str1)
tstp2, data2 = data_loader(fn, scale=False, ft_regex=regex_str2)
print(data1.shape, data2.shape)
ft_names = get_feature_names_bis(fn)[1:]
data1, ft_names1 = data_sanity(data1, ft_names, regex_str1)
data2, ft_names2 = data_sanity(data2, ft_names, regex_str2)

pos =[4800, 6000]
win = 300
assert tstp1 == tstp2
tstp_rel = [i - tstp1[0] for i in tstp1]
ft_ordered = []
for d in [data1, data2]:
    for p in pos:
        ft_ordered.append(sorted_ft(p, d, tstp_rel, win))

plot_gen(0, 2, ft_ordered)




# data_plots1 = plot_results([[fts[i][1] for i in range(len(fts))] for fts in ft_ordered[:2]], [list(range(len(ft_ordered[i]))) for i in range(2)],  ['enable/disable BFD session', 'enable/disable interface'], [[ft_names1[idx] for idx, _ in fts] for fts in ft_ordered[:2]])
# plot(data_plots1)
# data_plots2 = plot_results([[fts[i][1] for i in range(len(fts))] for fts in ft_ordered[2:]], [list(range(len(ft_ordered[i]))) for i in range(2,4)],  ['enable/disable BFD session', 'enable/disable interface'], [[ft_names2[idx] for idx, _ in fts] for fts in ft_ordered[2:]])
# plot(data_plots2)
d1 = [[[fts[i][1] for i in range(len(fts))] for fts in ft_ordered[0:2]], [list(range(len(ft_ordered[i]))) for i in range(2)], ['enable/disable BFD session', 'enable/disable interface']]
d2 = [[[fts[i][1] for i in range(len(fts))] for fts in ft_ordered[2:]], [list(range(len(ft_ordered[i]))) for i in range(2,4)], ['enable/disable BFD session', 'enable/disable interface']]
gen_plots(d1,[[ft_names1[idx] for idx, _ in fts] for fts in ft_ordered[0:2]], 'full-features')
gen_plots(d2,[[ft_names2[idx] for idx, _ in fts] for fts in ft_ordered[2:]], 'partial-features')
