import os
import pandas as pd
from insolver.feature_monitoring import HomogeneityReport, render_report
from insolver.model_tools.datasets import download_dataset
from sklearn.model_selection import train_test_split


download_dataset('US_Accidents_small')
df = pd.read_csv('./datasets/US_Accidents_small.csv')
train, test = train_test_split(df)

config = {
    'Temperature(F)': {
        "feature_type": 'continuous',
        "pval_thresh": 0.05,
        "samp_size": 500,
        "bootstrap_num": 100,
        "psi_bins": 10,
        "chart_bins": 50,
    },
    'Wind_Direction': {"feature_type": 'discrete', "pval_thresh": 0.05, "samp_size": 500, "bootstrap_num": 100},
}

hom_report = HomogeneityReport(config)
report_data = hom_report.build_report(train, test, name1='train', name2='test', draw_charts=True)
render_report(report_data)

os.remove('./datasets/US_Accidents_small.csv')
os.rmdir('./datasets/')

assert os.path.exists('homogeneity_report.html')
os.remove('homogeneity_report.html')
