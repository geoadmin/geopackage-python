geopackage-python : Python-based tools for creating OGC GeoPackages.
=================

[GeoPackage Specification from the Open Geospatial
Consortium](http://opengeospatial.org/standards/geopackage)

[![Build Status](https://travis-ci.org/GitHubRGI/geopackage-python.svg?branch=master)](https://travis-ci.org/GitHubRGI/geopackage-python)
[![Coverage Status](https://img.shields.io/coveralls/GitHubRGI/geopackage-python.svg)](https://coveralls.io/r/GitHubRGI/geopackage-python)
[![Stories in Ready](https://badge.waffle.io/GitHubRGI/geopackage-python.png?label=ready&title=Ready)](https://waffle.io/GitHubRGI/geopackage-python)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/GitHubRGI/geopackage-python/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/GitHubRGI/geopackage-python/?branch=master)

### Table Of Contents

* Installing Dependencies
  ([Windows](https://github.com/GitHubRGI/geopackage-python/wiki/Installing-dependencies-on-Windows), [Linux](https://github.com/GitHubRGI/geopackage-python/wiki/Installing-dependencies-on-Linux))
* [How to use the tiling script,
  gdal2tiles_parallel.py](https://github.com/GitHubRGI/geopackage-python/wiki/Usage-Instructions-for-gdal2tiles_parallel.py)
* [How to use the packaging script,
  tiles2gpkg_parallel.py](https://github.com/GitHubRGI/geopackage-python/wiki/Usage-Instructions-for-tiles2gpkg_parallel.py)
* [Running unit tests on
  tiles2gpkg_parallel.py](https://github.com/GitHubRGI/geopackage-python/wiki/Running-Unit-Tests-On-tiles2gpkg_parallel.py)

### In addition to the above documentation, the following features are added in this branch

* Added optional -maxlvl [number] parameter. Tiles level smaller or equal to this number will be packaged.
* Added optional -tm ['swiss_lv03' or 'swiss_esri_lv03' or swiss_esri_lv05] parameter. Enables custom non-regular tile matrix (eCH-0056 or custom ESRI Tile Cache).
* Added Swiss projection LV03 and LV95.
* Added ability to package ESRI Tile Cache (Only in "exploded" format).
* Added ability to package tiles stored in a s3 bucket. Use s3:// to package s3 directory.

###Examples
* First 7 levels of an exploded ArcGIS Cache, Google Map projection. 
python.exe tiles2gpkg_parallel.py -srs 3857 -maxlvl 7 -tileorigin ul \path_to_arcgis_cache\Layers\_alllayers geopackage.gpkg

* First 19 levels of an exploded ArcGIS Cache, Swiss LV95 projection.
python.exe tiles2gpkg_parallel.py -srs 2056 -tm swiss_lv95 -maxlvl 19 -tileorigin ul \path_to_arcgis_cache\Layers\_alllayers geopackage.gpkg

* First 21 levels of a WMTS whose tiles are stored in Amazon S3.
tiles2gpkg_parallel.py -maxlvl 21 -srs 2056 -tm swiss_lv95 -tileorigin ul s3://wts_instance_name/1.0.0/my_layer_name/default/teimestamp/21781 geopackage.gpkg
