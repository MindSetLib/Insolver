import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def download_dataset(name, folder='datasets'):
    """Function for downloading and unzipping example datasets

    Args:
        name (str): Dataset name. Available datasets are freMPL-R, US_Accidents, Lending_Club, weatherAus and
         AB_NYC_2019.
        folder (str): Path to the folder to dataset saving

    Returns:
        str: Information about saved dataset

    """
    datasets = {
        'freMPL-R': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/freMPL-R.zip',
        'US_Accidents': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/US_Accidents_June20.zip',
        'US_Accidents_small': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.5/US_Accidents_small.zip',
        'Lending_Club': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/LendingClub.zip',
        'weatherAUS': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.15/weatherAUS.zip',
        'AB_NYC_2019': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.15/AB_NYC_2019.zip',
    }
    if name not in datasets.keys():
        return f'Dataset {name} is not found. Available datasets are {", ".join(datasets.keys())}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    url = datasets[name]
    with urlopen(url) as file:
        with ZipFile(BytesIO(file.read())) as zfile:
            zfile.extractall(folder)

    return f'Dataset {name} saved to "{folder}" folder'
