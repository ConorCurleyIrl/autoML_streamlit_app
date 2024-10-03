
# 1. importing libraries - see requirements.txt for all libraries used
#######################################################################
import streamlit as st 

from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model, plot_model, evaluate_model, tune_model
import pandas as pd
import time
import pkg_resources

from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport


def get_package_versions(packages):
    versions = {}
    for package in packages:
        try:
            versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not installed"
    return versions

# List of packages to check
packages = [
    "streamlit",
    "pycaret",
    "pandas",
    "streamlit_pandas_profiling",
    "streamlit_option_menu",
    "ydata_profiling"
]

# Get package versions
package_versions = get_package_versions(packages)

# Print package versions to the terminal
print("Package Versions:")
for package, version in package_versions.items():
    print(f"{package}: {version}")
