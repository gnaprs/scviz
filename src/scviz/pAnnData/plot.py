

class PlotMixin:
    def plot_counts(self, classes=None, y='protein_count', **kwargs):
        import seaborn as sns

        df = self.summary # type: ignore #, in base
        if classes is None:
            df.reset_index()
            classes = 'index'
        sns.violinplot(data=df, x=classes, y=y, **kwargs)