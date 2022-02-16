from analysis_tools.common import *


## Missing value
def plot_missing_value(data, figsize=get_figsize(1, 1)):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    with FigProcessor(fig, PATH.RESULT, PLOT, "Missing value in full data"):
        msno.matrix(data, ax=axes[0])
        ms = data.isnull().sum()
        sns.barplot(ms.index, ms, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0])
        axes[1].set_xticklabels([])


## Feature exploration
### Single feature
def plot_num_feature(data_f, bins=BINS, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.histplot(data_f, bins=bins, ax=ax, kde=True, stat='density')
        ax.set_xlabel(None)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, data_f.name):
            plot_fn(ax)
def plot_cat_feature(data_f, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        density = data_f.value_counts().sort_index() / len(data_f)
        sns.barplot(density.index, density.values, ax=ax)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, data_f.name):
            plot_fn(ax)

### Multiple features
def plot_features(data, bins=BINS, n_cols=3, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    n_features = len(data.columns)
    n_rows     = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Feature distribution"):
        for ax, f in zip(axes.flat, data):
            ax.set_title(f)
            if data[f].nunique() > bins:
                ## Numerical feature or categorical feature
                try:
                    ax.hist(data[f], bins=bins, density=True, color='olive', alpha=0.5)
                except Exception as e:
                    print(f"[{f}]: {e}")
            else:
                ## Categorical feature
                cnts = data[f].value_counts().sort_index() / len(data[f])
                ax.bar(cnts.index, cnts.values, width=0.5, alpha=0.5)
                ax.set_xticks(cnts.index)
def plot_num_num_features(data, f1, f2, bins=BINS, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.histplot(x=data[f1], y=data[f2], bins=bins, ax=ax)
        ax.set_xlabel(None);  ax.set_ylabel(None)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, f"{f1} vs {f2}"):
            plot_fn(ax)
def plot_num_cat_features(data, f1, f2, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.violinplot(x=data[f1], y=data[f2], ax=ax, orient='h', order=sorted(data[f2].unique()), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, f"{f1} vs {f2}"):
            plot_fn(ax)
def plot_cat_num_features(data, f1, f2, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.violinplot(x=data[f1], y=data[f2], ax=ax, orient='v', order=sorted(data[f1].unique()), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, f"{f1} vs {f2}"):
            plot_fn(ax)
def plot_cat_cat_features(data, f1, f2, ax=None, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        ratio = pd.crosstab(data[f2], data[f1]) / len(data)
        ratio.sort_index(inplace=True)  # sort by index
        ratio = ratio[sorted(ratio)]    # sort by column
        ratio = ratio.iloc[:20, :20]    # limit: 20 x 20
        sns.heatmap(ratio, ax=ax, annot=True, fmt=".2f", cmap=sns.light_palette('gray', as_cmap=True), cbar=False)
        ax.set_xlabel(None);  ax.set_ylabel(None)

    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, f"{f1} vs {f2}"):
            plot_fn(ax)
def plot_corr(data, figsize=get_figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Correlation matrix"):
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)
