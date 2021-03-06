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

### In addition to the above documentation, the following features are added in this fork

* Added optional -maxlvl [number] parameter. Tiles level smaller or equal to this number will be packaged.
* Added optional -tm ['swiss_lv03', swiss_lv05] parameter. Enables custom non-regular tile matrix (eCH-0056 or custom ESRI Tile Cache).
* Added ability to package ESRI Exploded Tile Cache.
* The Script configures the Geopackage with the Conf.xml file from ESRI Tile Cache format automatically.
* Added ability to package tiles stored in a s3 bucket. Use s3:// to package s3 directory.
* Ability to give an initial extent for the content (for zoom to layer functions in common GIS).
* Added ability to package tiles stored in ESRI Compact Cache V2 Bundles.

###Examples
#### All levels of an exploded ArcGIS Cache (Script reads Conf.xml file from Cache).

* python.exe tiles2gpkg_parallel.py -initialextent minX,minY,maxX,maxY \path_to_arcgis_cache\Layers\_alllayers geopackage.gpkg

#### First 19 levels of an exploded ArcGIS Cache (Script reads Conf.xml file from Cache).

* python.exe tiles2gpkg_parallel.py -maxlvl 19 -initialextent minX,minY,maxX,maxY \path_to_arcgis_cache\Layers\_alllayers geopackage.gpkg

#### First 21 levels of a WMTS whose tiles are stored in Amazon S3 + custom initial extent.

* tiles2gpkg_parallel.py -initialextent 2375000,980000,2998000,1421000 -maxlvl 21 -srs 2056 -tm swiss_lv95 -tileorigin ul s3://wts_instance_name/1.0.0/my_layer_name/default/timestamp/21781 geopackage.gpkg
