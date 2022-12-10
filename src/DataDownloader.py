""".
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")

#import os
#import pandas as pd
#import numpy as np
from pathlib import Path
import requests
from zipfile import ZipFile
from io import BytesIO
from zippeekiyay import namelist


class DataDownloader:

    @staticmethod
    def flist_exists(flist: list, path: Path = None) -> bool:
        if path is not None:
            # append file names to given path
            flist = [Path(path, f) for f in flist]
        else:
            # turn str into proper Path objects
            flist = [Path(f) for f in flist]

        # check if its either file or directory
        for f in flist:
            if not (f.is_file() or f.is_dir()):
                return False

        return True


    @staticmethod
    def _dl_and_unpack(url: str, dst: Path) -> ZipFile:
        # download into memory
        req = requests.get(url)

        # create archive and unzip
        zfile = ZipFile(BytesIO(req.content))
        zfile.extractall(dst)

        return zfile

    
    @staticmethod
    def download_and_unpack(url: str,
                            dst: Path,
                            cache=True,
                            double_check=False) -> bool:
        flist = None

        if cache:
            # get list of files in archive without downloading it
            flist = namelist(url)

            # download only if not done already
            if not DataDownloader.flist_exists(flist, dst):
                zfile = DataDownloader._dl_and_unpack(url, dst)
                flist = zfile.namelist()
        else:
            zfile = DataDownloader._dl_and_unpack(url, dst)
            flist = zfile.namelist()

        result = True
        if double_check:
            result = DataDownloader.flist_exists(flist, dst)

        return result
