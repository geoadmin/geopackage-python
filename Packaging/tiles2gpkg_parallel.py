#!/usr/bin/python2.7
"""
Copyright (C) 2014 Reinventing Geospatial, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>,
or write to the Free Software Foundation, Inc., 59 Temple Place -
Suite 330, Boston, MA 02111-1307, USA.

Author: Steven D. Lander, Reinventing Geospatial Inc (RGi)
Date: 2013-07-12
   Requires: sqlite3, argparse
   Optional: Python Imaging Library (PIL or Pillow)
Description: Converts a TMS folder into a geopackage with
 PNGs for images with transparency and JPEGs for those
 without.
Credits:
  MapProxy imaging functions: http://mapproxy.org
  gdal2mb on github: https://github.com/developmentseed/gdal2mb

Version:
"""

from glob import glob
try:
    from cStringIO import StringIO as ioBuffer
except ImportError:
    from io import BytesIO as ioBuffer
from time import sleep
from uuid import uuid4
from sys import stdout
from sys import version_info
if version_info[0] == 3:
    xrange = range
from operator import attrgetter
from sqlite3 import connect, Error
from argparse import ArgumentParser
from sqlite3 import Binary as sbinary
from os import walk, remove
from os.path import split, join, exists
from multiprocessing import cpu_count, Pool
from math import pi, sin, log, tan, atan, sinh, degrees
import math
import os
import re
try:
    from PIL.Image import open as IOPEN
except ImportError:
    IOPEN = None

# JPEGs @ 75% provide good quality images with low footprint, use as a default
# PNGs should be used sparingly (mixed mode) due to their high disk usage RGBA
# Options are mixed, jpeg, and png
IMAGE_TYPES = '.png', '.jpeg', '.jpg'

class GPSConverter(object):
    '''
    GPS Converter class which is able to perform convertions between the
    CH1903 and WGS84 system.
    '''
    # Convert CH y/x/h to WGS height
    def CHtoWGSheight(self, y, x, h):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        h = (h + 49.55) - (12.60 * y_aux) - (22.64 * x_aux)
        return h

    # Convert CH y/x to WGS lat
    def CHtoWGSlat(self, y, x):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lat = (16.9023892 + (3.238272 * x_aux)) + \
                - (0.270978 * pow(y_aux, 2)) + \
                - (0.002528 * pow(x_aux, 2)) + \
                - (0.0447 * pow(y_aux, 2) * x_aux) + \
                - (0.0140 * pow(x_aux, 3))
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lat = (lat * 100) / 36
        return lat

    # Convert CH y/x to WGS long
    def CHtoWGSlng(self, y, x):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lng = (2.6779094 + (4.728982 * y_aux) + \
                + (0.791484 * y_aux * x_aux) + \
                + (0.1306 * y_aux * pow(x_aux, 2))) + \
                - (0.0436 * pow(y_aux, 3))
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lng = (lng * 100) / 36
        return lng

    # Convert decimal angle (? dec) to sexagesimal angle (dd.mmss,ss)
    def DecToSexAngle(self, dec):
        degree = int(math.floor(dec))
        minute = int(math.floor((dec - degree) * 60))
        second = (((dec - degree) * 60) - minute) * 60
        return degree + (float(minute) / 100) + (second / 10000)

    # Convert sexagesimal angle (dd.mmss,ss) to seconds
    def SexAngleToSeconds(self, dms):
        degree = 0
        minute = 0
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return second + (minute * 60) + (degree * 3600)

    # Convert sexagesimal angle (dd.mmss) to decimal angle (degrees)
    def SexToDecAngle(self, dms):
        degree = 0
        minute = 0
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return degree + (minute / 60) + (second / 3600)

    # Convert WGS lat/long (? dec) and height to CH h
    def WGStoCHh(self, lat, lng, h):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        h = (h - 49.55) + (2.73 * lng_aux) + (6.94 * lat_aux)
        return h

    # Convert WGS lat/long (? dec) to CH x
    def WGStoCHx(self, lat, lng):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        x = ((200147.07 + (308807.95 * lat_aux) + \
            + (3745.25 * pow(lng_aux, 2)) + \
            + (76.63 * pow(lat_aux,2))) + \
            - (194.56 * pow(lng_aux, 2) * lat_aux)) + \
            + (119.79 * pow(lat_aux, 3))
        return x

	# Convert WGS lat/long (? dec) to CH y
    def WGStoCHy(self, lat, lng):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        y = (600072.37 + (211455.93 * lng_aux)) + \
            - (10938.51 * lng_aux * lat_aux) + \
            - (0.36 * lng_aux * pow(lat_aux, 2)) + \
            - (44.54 * pow(lng_aux, 3))
        return y

    def LV03toWGS84(self, east, north, height):
        '''
        Convert LV03 to WGS84 Return a array of double that contain lat, long,
        and height
        '''
        d = []
        d.append(self.CHtoWGSlat(east, north))
        d.append(self.CHtoWGSlng(east, north))
        d.append(self.CHtoWGSheight(east, north, height))
        return d

    def WGS84toLV03(self, latitude, longitude, ellHeight):
        '''
        Convert WGS84 to LV03 Return an array of double that contaign east,
        north, and height
        '''
        d = []
        d.append(self.WGStoCHy(latitude, longitude))
        d.append(self.WGStoCHx(latitude, longitude))
        d.append(self.WGStoCHh(latitude, longitude, ellHeight))
        return d

class Mercator(object):
    """
    Mercator projection class that holds specific calculations and formulas
    for EPSG3857.
    """

    def __init__(self, tile_size=256):
        """
        Constructor
        """
        self.tile_size = tile_size
        self.radius = 6378137
        self.origin_shift = pi * self.radius
        self.initial_resolution = 2 * self.origin_shift / self.tile_size

    @staticmethod
    def invert_y(z, y):
        """
        Inverts the Y tile value.

        Inputs:
        z -- the zoom level associated with the tile
        y -- the Y tile number

        Returns:
        The flipped tile value
        """
        return (1 << z) - y - 1

    @staticmethod
    def tile_to_lat_lon(z, x, y):
        """
        Returns the lat/lon coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        n = 2.0 ** z
        lon = x / n * 360.0 - 180.0
        lat_rad = atan(sinh(pi * (2 * y / n - 1)))
        #lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = degrees(lat_rad)
        return lat, lon

    def tile_to_meters(self, z, x, y):
        """
        Returns the meter coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        # Mercator Upper left, add 1 to both x and y to get Lower right
        lat, lon = self.tile_to_lat_lon(z, x, y)
        meters_x = lon * self.origin_shift / 180.0
        meters_y = log(tan((90 + lat) * pi / 360.0)) / \
                                                            (pi / 180.0)
        meters_y = meters_y * self.origin_shift / 180.0
        return meters_x, meters_y

    @staticmethod
    def pixel_size(z):
        """
        Returns the pixel resolution of the input zoom level.

        Inputs:
        z -- zoom level value for the input tile
        """
        return 156543.033928041 / 2**z

    def get_coord(self, z, x, y):
        """
        Returns the coordinates (in meters) of the bottom-left corner of the
        input tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        return self.tile_to_meters(z, x, y)

    @staticmethod
    def truncate(coord):
        """
        Formats a coordinate to within an acceptable degree of accuracy (2
        decimal places for mercator).
        """
        return '%.2f' % (int(coord*100)/float(100))


class Geodetic(object):
    """
    Geodetic projection class that holds specific calculations and formulas for
    EPSG4326.
    """
    def __init__(self, tile_size=256):
        """
        Constructor
        """
        self.tile_size = tile_size
        self.resolution_factor = 360.0 / self.tile_size

    def pixel_size(self, zoom):
        """
        Return the size of a pixel in lat/long at the given zoom level

        z -- zoom level of the tile
        """
        return self.resolution_factor / 2 ** zoom

    def get_coord(self, z, x, y):
        """
        Return the coordinates (in lat/long) of the bottom left corner of
        the tile

        z -- zoom level for input tile
        x -- tile column
        y -- tile row
        """
        res = self.resolution_factor / 2**z
        return x*self.tile_size*res -180, y*self.tile_size*res - 90

    @staticmethod
    def invert_y(z, y):
        """
        Return the inverted Y value of the tile

        z -- zoom level
        """
        if z == 0:
            return 0
        else:
            return (1 << (z - 1)) - y - 1

    @staticmethod
    def truncate(coord):
        """
        Formats a coordinate to an acceptable degree of accuracy (7 decimal
        places for Geodetic).
        """
        return '%.7f' % (int(coord*10000000)/float(10000000))


class EllipsoidalMercator(Mercator):
    """
    Ellipsoidal Mercator projection class that holds specific calculations and
    formulas for EPSG3395.
    """
    def __init__(self):
        """
        Constructor
        """
        super(EllipsoidalMercator, self).__init__()

    @staticmethod
    def lat_to_northing(lat):
        """
        Convert a latitude to a northing
                      /    / pi   phi \   / 1 - e sin(phi) \ e/2 \
        y(phi) = R ln| tan| --- + ---  | |  --------------  |     |
                      \    \ 4     2  /   \ 1 + e sin(phi) /     /
        """
        r = 6378137.0
        e = 0.081819190842621
        return r * log(tan((pi/2 + lat) / 2) *
                    ((1 - e * sin(lat))/(1 + e * sin(lat))) ** (e/2))

    @staticmethod
    def tile_to_lat_lon(z, x, y):
        """
        Returns the lat/lon coordinates of the bottom-left corner of the input
        tile. Finds the value numerically (using the secant method).

        Inputs:
        z -- zoom level value for input tile
        x -- tile column value for input tile
        y -- tile row value for input tile
        """
        n = 2.0 ** z
        lon = x / n * 360.0 - 180.0
        my = (y - 2 ** (z - 1)) * 6378137 * pi * 2 / 2 ** z
        def f(phi): return EllipsoidalMercator.lat_to_northing(phi) - my
        lat = 0.0
        oldLat = 1.0
        diff = 1.0
        while abs(diff) > 0.0001:
            newLat = lat - f(lat) * (lat - oldLat) / (f(lat) - f(oldLat))
            if newLat > 1.48499697138:
                newLat = 1.48499697138
            elif newLat < -1.48499697138:
                newLat = -1.48499697138
            oldLat = lat
            lat = newLat
            diff = lat - oldLat
        lat = lat * 180.0 / pi
        return lat, lon

    def tile_to_meters(self, z, x, y):
        """
        Returns the meter coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        lat, lon = self.tile_to_lat_lon(z, x, y)
        meters_x = lon * self.origin_shift / 180.0
        meters_y = self.lat_to_northing(lat * pi / 180.0)
        return meters_x, meters_y


class ScaledWorldMercator(EllipsoidalMercator):
    """
    Scaled World Mercator projection class that holds specific calculations
    and formulas for EPSG9804/9805 projection proposed by NGA Craig Rollins.
    """
    def __init__(self):
        """
        Constructor
        """
        super(ScaledWorldMercator, self).__init__()

    @staticmethod
    def pixel_size(z):
        """
        Calculates the pixel size for a given zoom level.
        """
        return 125829.12 / 2**z

    @staticmethod
    def lat_to_northing(lat):
        """
        Convert a latitude to a northing
                      /    / pi   phi \   / 1 - e sin(phi) \ e/2 \
        y(phi) = R ln| tan| --- + ---  | |  --------------  |     |
                      \    \ 4     2  /   \ 1 + e sin(phi) /     /
        """
        r = 6378137.0 * 0.857385503731176
        e = 0.081819190842621
        return r * log(tan((pi/2 + lat) / 2) *
                    ((1 - e * sin(lat))/(1 + e * sin(lat))) ** (e/2))

    @staticmethod
    def tile_to_lat_lon(z, x, y):
        """
        Returns the lat/lon coordinates of the bottom-left corner of the input
        tile. Finds the value numerically (using the secant method). A scale
        factor has been added specifically for scaled world mercator.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column value for input tile
        y -- tile row value for input tile
        """
        n = 2.0 ** z
        r = 6378137.0 * 0.857385503731176
        lon = x / n * 360.0 - 180.0
        my = (y - 2 ** (z - 1)) * r * pi * 2 / 2 ** z
        def f(phi): return ScaledWorldMercator.lat_to_northing(phi) - my
        lat = 0.0
        oldLat = 1.0
        diff = 1.0
        while abs(diff) > 0.0001:
            newLat = lat - f(lat) * (lat - oldLat) / (f(lat) - f(oldLat))
            if newLat > 1.4849969713855238:
                newLat = 1.4849969713855238
            elif newLat < -1.4849969713855238:
                newLat = -1.4849969713855238
            oldLat = lat
            lat = newLat
            diff = lat - oldLat
        lat = lat * 180.0 / pi
        return lat, lon

    def tile_to_meters(self, z, x, y):
        """
        Returns the meter coordinates of the bottom-left corner of the input
        tile. A scale factor has been added to the longitude meters
        calculation.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        lat, lon = self.tile_to_lat_lon(z, x, y)
        meters_x = lon * (pi * (6378137.0 * 0.857385503731176)) / 180.0
        meters_y = self.lat_to_northing(lat * pi / 180.0)
        # Instituting a 2 decimal place round to ensure accuracy
        return meters_x, round(meters_y, 2)

class SwissTM_LV03:
    tile_size = 256
    top_left = [420000.0, 350000.0]
    scales = [[14285750.5715, [1,1]],
                                  [13392891.1608, [1,1]],[12500031.7501, [1,1]],[11607172.3393, [1,1]],
                                  [10714312.9286, [1,1]],[9821453.51791, [1,1]],[8928594.10719, [1,1]],
                                  [8035734.69647, [1,1]],[7142875.28575, [1,1]],[6250015.87503, [2,1]],
                                  [5357156.46431, [2,1]],[4464297.05359, [2,1]],[3571437.64288, [2,2]],
                                  [2678578.23216, [3,2]],[2321434.46787, [3,2]],[1785718.82144, [4,3]],
                                  [892859.410719, [8,5]],[357143.764288, [19,13]],[178571.882144, [38,25]],
                                  [71428.7528575, [94,63]],[35714.3764288, [188,125]],[17857.1882144, [375,250]],
                                  [8928.59410719, [750,500]],[7142.87528575, [938,625]],[5357.15646431, [1250,834]],
                                  [3571.43764288, [1875,1250]],[1785.71882144, [3750,2500]], [892.857, [7500,5000]], [357.1425, [18750,12500]]]

class SwissTM_LV95:
    tile_size = 256
    top_left = [2420000.0, 1350000.0]
    scales = [[14285750.5715, [1,1]],
                                  [13392891.1608, [1,1]],[12500031.7501, [1,1]],[11607172.3393, [1,1]],
                                  [10714312.9286, [1,1]],[9821453.51791, [1,1]],[8928594.10719, [1,1]],
                                  [8035734.69647, [1,1]],[7142875.28575, [1,1]],[6250015.87503, [2,1]],
                                  [5357156.46431, [2,1]],[4464297.05359, [2,1]],[3571437.64288, [2,2]],
                                  [2678578.23216, [3,2]],[2321434.46787, [3,2]],[1785718.82144, [4,3]],
                                  [892859.410719, [8,5]],[357143.764288, [19,13]],[178571.882144, [38,25]],
                                  [71428.7528575, [94,63]],[35714.3764288, [188,125]],[17857.1882144, [375,250]],
                                  [8928.59410719, [750,500]],[7142.87528575, [938,625]],[5357.15646431, [1250,834]],
                                  [3571.43764288, [1875,1250]],[1785.71882144, [3750,2500]], [892.857, [7500,5000]], [357.1425, [18750,12500]]]


    #WMTS Standard is 25.4mm per inch / 0.28mm per pixel(dot) = 90.71 dot per inch
    stdRdrPixelSize=0.00028

class SwissTM_ESRI_LV03:
    tile_size = 256
    top_left = [-29386400, 30814500]
    scales =   [[8000000, [55+1,56+1]],
                                 [4000000, [110+2,112+2]],
                                  [2000000, [220+4,225+3]],
                                  [1000000, [440+7,450+5]],
                                  [500000, [880+13,900+9]],
                                  [250000, [1761+25,1801+16]],
                                  [125000, [3522+50,3603+31]]]
    # ESRI Standard is 96 dot per inch, so a pixel size of: 0.264583 mm
    stdRdrPixelSize=0.000264583862501058376


class CH1903LV03(object):
    """
    Mercator projection class that holds specific calculations and formulas
    for EPSG21781.
    """

    converter = GPSConverter()

    def __init__(self):
        """
        Constructor
        https://api3.geo.admin.ch/1.0.0/WMTSCapabilities.xml
        https://gist.github.com/atlefren/c41921d64a2636c9598e
        """
        self.tile_matrix = None

    def setTileMatrix(self, tile_matrix):
        self.tile_matrix = tile_matrix
        self.tile_size = tile_matrix.tile_size

    @staticmethod
    def invert_y(z, y):
        """
        Inverts the Y tile value.

        Inputs:
        z -- the zoom level associated with the tile
        y -- the Y tile number

        Returns:
        The flipped tile value
        """
        return (1 << z) - y - 1

    @staticmethod
    def tile_to_lat_lon(z, x, y):
        """
        Returns the lat/lon coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        meters_x = this.tile_matrix.top_left[0] + x * this.tile_matrix.tile_size * this.tile_matrix.stdRdrPixelSize * this.tile_matrix.scales[z][0]
        meters_y = this.tile_matrix.top_left[1] - (y + 1) * this.tile_matrix.tile_size * this.tile_matrix.stdRdrPixelSize * this.tile_matrix.scales[z][0]

        lat = CH1903LV03.converter.CHtoWGSlat(meters_x, meters_y)
        lon = CH1903LV03.converter.CHtoWGSlng(meters_x, meters_y)

        return lat, lon

    def tile_to_meters(self, z, x, y):
        """
        Returns the meter coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """

        # Mercator Upper left, add 1 to both x and y to get Lower right
        meters_x = self.tile_matrix.top_left[0] + x * self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0]
        meters_y = self.tile_matrix.top_left[1] - (y + 1) * self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0]

        return meters_x, meters_y

    #@staticmethod
    def pixel_size(self, z):
        """
        Returns the pixel resolution of the input zoom level.

        Inputs:
        z -- zoom level value for the input tile
        """
        return self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0] / self.tile_matrix.tile_size

    def get_coord(self, z, x, y):
        """
        Returns the coordinates (in meters) of the bottom-left corner of the
        input tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        return self.tile_to_meters(z, x, y)

    @staticmethod
    def truncate(coord):
        """
        Formats a coordinate to within an acceptable degree of accuracy (2
        decimal places for mercator).
        """
        return '%.2f' % (int(coord*100)/float(100))

class CH1903pLV95(object):
    """
    Mercator projection class that holds specific calculations and formulas
    for EPSG2056.
    """

    converter = GPSConverter()

    def __init__(self):
        """
        Constructor
        https://api3.geo.admin.ch/1.0.0/WMTSCapabilities.xml
        https://gist.github.com/atlefren/c41921d64a2636c9598e
        """
        self.tile_matrix = None

    def setTileMatrix(self, tile_matrix):
        self.tile_matrix = tile_matrix
        self.tile_size = tile_matrix.tile_size

    @staticmethod
    def invert_y(z, y):
        """
        Inverts the Y tile value.

        Inputs:
        z -- the zoom level associated with the tile
        y -- the Y tile number

        Returns:
        The flipped tile value
        """
        return (1 << z) - y - 1

    @staticmethod
    def tile_to_lat_lon(z, x, y):
        """
        Returns the lat/lon coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        meters_x = this.tile_matrix.top_left[0] + x * this.tile_matrix.tile_size * this.tile_matrix.stdRdrPixelSize * this.tile_matrix.scales[z][0] - 2000000
        meters_y = this.tile_matrix.top_left[1] - (y + 1) * this.tile_matrix.tile_size * this.tile_matrix.stdRdrPixelSize * this.tile_matrix.scales[z][0] - 1000000

        lat = CH1903LV03.converter.CHtoWGSlat(meters_x, meters_y)
        lon = CH1903LV03.converter.CHtoWGSlng(meters_x, meters_y)

        return lat, lon

    def tile_to_meters(self, z, x, y):
        """
        Returns the meter coordinates of the bottom-left corner of the input
        tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """

        # Mercator Upper left, add 1 to both x and y to get Lower right
        meters_x = self.tile_matrix.top_left[0] + x * self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0]
        meters_y = self.tile_matrix.top_left[1] - (y + 1) * self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0]

        return meters_x, meters_y


    #@staticmethod
    def pixel_size(self, z):
        """
        Returns the pixel resolution of the input zoom level.

        Inputs:
        z -- zoom level value for the input tile
        """
        return self.tile_matrix.tile_size * self.tile_matrix.stdRdrPixelSize * self.tile_matrix.scales[z][0] / self.tile_matrix.tile_size

    def get_coord(self, z, x, y):
        """
        Returns the coordinates (in meters) of the bottom-left corner of the
        input tile.

        Inputs:
        z -- zoom level value for input tile
        x -- tile column (longitude) value for input tile
        y -- tile row (latitude) value for input tile
        """
        return self.tile_to_meters(z, x, y)

    @staticmethod
    def truncate(coord):
        """
        Formats a coordinate to within an acceptable degree of accuracy (2
        decimal places for mercator).
        """
        return '%.2f' % (int(coord*100)/float(100))


class ZoomMetadata(object):

    """Return an object containing metadata about a given zoom level."""

    @property
    def zoom(self):
        """Return the zoom level of this metadata object."""
        return self.__zoom

    @zoom.setter
    def zoom(self, value):
        """Set the zoom level of this metadata object."""
        self.__zoom = value

    @property
    def min_tile_col(self):
        """Return the minimum tile column of this metadata object."""
        return self.__min_tile_col

    @min_tile_col.setter
    def min_tile_col(self, value):
        """Set the minimum tile column of this metadata object."""
        self.__min_tile_col = value

    @property
    def max_tile_col(self):
        """Return the maximum tile column of this metadata object."""
        return self.__max_tile_col

    @max_tile_col.setter
    def max_tile_col(self, value):
        """Set the maximum tile column of this metadata object."""
        self.__max_tile_col = value

    @property
    def min_tile_row(self):
        """Return the minimum tile row of this metadata object."""
        return self.__min_tile_row

    @min_tile_row.setter
    def min_tile_row(self, value):
        """Set the minimum tile row of this metadata object."""
        self.__min_tile_row = value

    @property
    def max_tile_row(self):
        """Return the maximum tile row of this metadata object."""
        return self.__max_tile_row

    @max_tile_row.setter
    def max_tile_row(self, value):
        """Set the maximum tile row of this metadata object."""
        self.__max_tile_row = value

    @property
    def min_x(self):
        """Return the minimum x coordinate of the bounding box."""
        return self.__min_x

    @min_x.setter
    def min_x(self, value):
        """Set the minimum x coordinate of the bounding box."""
        self.__min_x = value

    @property
    def max_x(self):
        """Return the maximum x coordinate of the bounding box."""
        return self.__max_x

    @max_x.setter
    def max_x(self, value):
        """Set the maximum x coordinate of the bounding box."""
        self.__max_x = value

    @property
    def min_y(self):
        """Return the minimum y coordinate of the bounding box."""
        return self.__min_y

    @min_y.setter
    def min_y(self, value):
        """Set the minimum y coordinate of the bounding box."""
        self.__min_y = value

    @property
    def max_y(self):
        """Return the maximum y coordinate of the bounding box."""
        return self.__max_y

    @max_y.setter
    def max_y(self, value):
        """Set the maximum y coordinate of the bounding box."""
        self.__max_y = value

    @property
    def matrix_width(self):
        """Number of tiles wide this matrix should be."""
        #return (self.__matrix_width if hasattr(self, 'matrix_width') else None)
        return self.__matrix_width or None

    @matrix_width.setter
    def matrix_width(self, value):
        """Set the number of tiles wide this matrix should be."""
        self.__matrix_width = value

    @property
    def matrix_height(self):
        """Number of tiles high this matrix should be."""
        return self.__matrix_height or None

    @matrix_height.setter
    def matrix_height(self, value):
        """Set the number of tiles high this matrix should be."""
        self.__matrix_height = value


class Geopackage(object):

    def __enter__(self):
        """With-statement caller"""
        return self

    def __init__(self, file_path, srs, tile_matrix):
        """Constructor."""
        self.__file_path = file_path
        self.__srs = srs
        if self.__srs == 3857:
            self.__projection = Mercator()
        elif self.__srs == 3395:
            self.__projection = EllipsoidalMercator()
        elif self.__srs == 9804:
            self.__projection = ScaledWorldMercator()
        elif self.__srs == 21781:
            self.__projection = CH1903LV03()
        elif self.__srs == 2056:
            self.__projection = CH1903pLV95()
        else:
            self.__projection = Geodetic()
        self.__db_con = connect(self.__file_path)
        self.__create_schema()

        #setup tile matrix if specified
        if tile_matrix == "swiss_lv03":
            self.__projection.setTileMatrix(SwissTM_LV03())
        elif tile_matrix == "swiss_lv95":
            self.__projection.setTileMatrix(SwissTM_LV95())
        elif tile_matrix == "swiss_esri_lv03":
            self.__projection.setTileMatrix(SwissTM_ESRI_LV03())

    def __create_schema(self):
        """Create default geopackage schema on the database."""
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            cursor.execute("""
                CREATE TABLE gpkg_contents (
                    table_name TEXT NOT NULL PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    identifier TEXT UNIQUE,
                    description TEXT DEFAULT '',
                    last_change DATETIME NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    min_x DOUBLE,
                    min_y DOUBLE,
                    max_x DOUBLE,
                    max_y DOUBLE,
                    srs_id INTEGER,
                    CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                        REFERENCES gpkg_spatial_ref_sys(srs_id));
            """)
            cursor.execute("""
                CREATE TABLE gpkg_spatial_ref_sys (
                    srs_name TEXT NOT NULL,
                    srs_id INTEGER NOT NULL PRIMARY KEY,
                    organization TEXT NOT NULL,
                    organization_coordsys_id INTEGER NOT NULL,
                    definition TEXT NOT NULL,
                    description TEXT);
            """)
            cursor.execute("""
                CREATE TABLE gpkg_tile_matrix (
                    table_name TEXT NOT NULL,
                    zoom_level INTEGER NOT NULL,
                    matrix_width INTEGER NOT NULL,
                    matrix_height INTEGER NOT NULL,
                    tile_width INTEGER NOT NULL,
                    tile_height INTEGER NOT NULL,
                    pixel_x_size DOUBLE NOT NULL,
                    pixel_y_size DOUBLE NOT NULL,
                    CONSTRAINT pk_ttm PRIMARY KEY (table_name, zoom_level),
                    CONSTRAINT fk_ttm_table_name FOREIGN KEY (table_name)
                        REFERENCES gpkg_contents(table_name));
            """)
            cursor.execute("""
                CREATE TABLE gpkg_tile_matrix_set (
                    table_name TEXT NOT NULL PRIMARY KEY,
                    srs_id INTEGER NOT NULL,
                    min_x DOUBLE NOT NULL,
                    min_y DOUBLE NOT NULL,
                    max_x DOUBLE NOT NULL,
                    max_y DOUBLE NOT NULL,
                    CONSTRAINT fk_gtms_table_name FOREIGN KEY (table_name)
                        REFERENCES gpkg_contents(table_name),
                    CONSTRAINT fk_gtms_srs FOREIGN KEY (srs_id)
                        REFERENCES gpkg_spatial_ref_sys(srs_id));
            """)
            cursor.execute("""
                CREATE TABLE tiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zoom_level INTEGER NOT NULL,
                    tile_column INTEGER NOT NULL,
                    tile_row INTEGER NOT NULL,
                    tile_data BLOB NOT NULL,
                    UNIQUE (zoom_level, tile_column, tile_row));
            """)
            cursor.execute("pragma foreign_keys = 1;")
            # Insert EPSG values for tiles table
            wkt = """
                PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984"
                ,SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]]
                ,AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG",
                "8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]
                ,AUTHORITY["EPSG","9122"]]AUTHORITY["EPSG","4326"]],PROJECTION[
                "Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER[
                "scale_factor",1],PARAMETER["false_easting",0],PARAMETER[
                "false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS[
                "X",EAST],AXIS["Y",NORTH]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (3857, ?, 3857, ?, ?)
            """, ("epsg", "WGS 84 / Pseudo-Mercator", wkt))
            wkt = """
                GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,
                298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG",
                "6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT
                ["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (4326, ?, 4326, ?, ?)
            """, ("epsg", "WGS 84", wkt))
            wkt = """
                PROJCS["WGS 84 / World Mercator",GEOGCS["WGS 84",
                DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],
                PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],
                PARAMETER["scale_factor",1],PARAMETER["false_easting",0],
                PARAMETER["false_northing",0],AUTHORITY["EPSG","3395"],
                AXIS["Easting",EAST],AXIS["Northing",NORTH]]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (3395, ?, 3395, ?, ?)
            """, ("epsg", "WGS 84 / World Mercator", wkt))

            wkt = """
                PROJCS["CH1903 / LV03",GEOGCS["CH1903",
                DATUM["CH1903",SPHEROID["Bessel 1841",6377397.155,299.1528128,
                AUTHORITY["EPSG","7004"]],TOWGS84[674.374,15.056,405.346,0,0,0,0],
                AUTHORITY["EPSG","6149"]],
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4149"]],
                UNIT["metre",1,AUTHORITY["EPSG","9001"]],
                PROJECTION["Hotine_Oblique_Mercator"],
                PARAMETER["latitude_of_center",46.95240555555556],
                PARAMETER["longitude_of_center",7.439583333333333],
                PARAMETER["azimuth",90],
                PARAMETER["rectified_grid_angle",90],
                PARAMETER["scale_factor",1],
                PARAMETER["false_easting",600000],
                PARAMETER["false_northing",200000],
                AUTHORITY["EPSG","21781"],
                AXIS["Y",EAST],AXIS["X",NORTH]]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (21781, ?, 21781, ?, ?)
            """, ("epsg", "CH1903 / LV03 -- Swiss CH1903 / LV03", wkt))

            wkt = """
                PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",
                DATUM["CH1903",SPHEROID["Bessel 1841",6377397.155,299.1528128,
                AUTHORITY["EPSG","7004"]],TOWGS84[674.374,15.056,405.346,0,0,0,0],
                AUTHORITY["EPSG","6150"]],
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4150"]],
                UNIT["metre",1,AUTHORITY["EPSG","9001"]],
                PROJECTION["Hotine_Oblique_Mercator"],
                PARAMETER["latitude_of_center",46.95240555555556],
                PARAMETER["longitude_of_center",7.439583333333333],
                PARAMETER["azimuth",90],PARAMETER["rectified_grid_angle",90],
                PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],
                PARAMETER["false_northing",1200000],AUTHORITY["EPSG","2056"],
                AXIS["Y",EAST],AXIS["X",NORTH]]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (2056, ?, 2056, ?, ?)
            """, ("epsg", "CH1903+ / LV95 -- Swiss CH1903+ / LV95", wkt))

            wkt = """
                PROJCS["unnamed",GEOGCS["WGS 84",
                DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],
                AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],
                PARAMETER["central_meridian",0],
                PARAMETER["scale_factor",0.803798909747978],
                PARAMETER["false_easting",0],
                PARAMETER["false_northing",0],
                UNIT["metre",1,AUTHORITY["EPSG","9001"]]]
            """
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (9804, ?, 9804, ?, ?)
            """, ("epsg", "WGS 84 / Scaled World Mercator", wkt))
            wkt = """undefined"""
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (-1, ?, -1, ?, ?)
            """, ("NONE", " ", wkt))
            cursor.execute("""
                INSERT INTO gpkg_spatial_ref_sys (
                    srs_id,
                    organization,
                    organization_coordsys_id,
                    srs_name,
                    definition)
                VALUES (0, ?, 0, ?, ?)
            """, ("NONE", " ", wkt))
            cursor.execute("""
                INSERT INTO gpkg_contents (
                    table_name,
                    data_type,
                    identifier,
                    description,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    srs_id)
                VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?);
            """, ("tiles", "tiles", "Raster Tiles",
                    "Created by tiles2gpkg_parallel.py, written by S. Lander",
                    self.__srs))
            # Add GP10 to the Sqlite header
            cursor.execute("pragma application_id = 1196437808;")

    @property
    def file_path(self):
        """Return the path of the geopackage database on the file system."""
        return self.__file_path

    def update_metadata(self, metadata):
        """Update the metadata of the geopackage database after tile merge."""
        # initialize a new projection
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            tile_matrix_stmt = """
                    INSERT OR REPLACE INTO gpkg_tile_matrix (
                        table_name,
                        zoom_level,
                        matrix_width,
                        matrix_height,
                        tile_width,
                        tile_height,
                        pixel_x_size,
                        pixel_y_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            # iterate through each zoom level object and assign
            # matrix data to table
            for level in metadata:
                cursor.execute(tile_matrix_stmt, (
                    "tiles", level.zoom, level.matrix_width, level.matrix_height,
                    self.__projection.tile_size, self.__projection.tile_size,
                    self.__projection.pixel_size(level.zoom),
                    self.__projection.pixel_size(level.zoom)))
            contents_stmt = """
                UPDATE gpkg_contents SET
                    min_x = ?,
                    min_y = ?,
                    max_x = ?,
                    max_y = ?
                WHERE table_name = 'tiles';
            """
            tile_matrix_set_stmt = """
                INSERT OR REPLACE INTO gpkg_tile_matrix_set (
                    table_name,
                    srs_id,
                    min_x,
                    min_y,
                    max_x,
                    max_y)
                VALUES (?, ?, ?, ?, ?, ?);
            """
            # get bounding box info based on
            top_level = min(metadata, key=attrgetter('zoom'))
            #top_level.min_x = self.__projection.truncate(top_level.min_x)
            #top_level.min_y = self.__projection.truncate(top_level.min_y)
            #top_level.max_x = self.__projection.truncate(top_level.max_x)
            #top_level.max_y = self.__projection.truncate(top_level.max_y)
            #
            #What is this?!?
            #
            top_level.min_x = top_level.min_x
            top_level.min_y = top_level.min_y
            top_level.max_x = top_level.max_x
            top_level.max_y = top_level.max_y
            # write bounds and matrix set info to table
            cursor.execute(contents_stmt, (top_level.min_x, top_level.min_y,
                top_level.max_x, top_level.max_y))
            cursor.execute(tile_matrix_set_stmt, ('tiles', self.__srs, top_level.min_x,
                top_level.min_y, top_level.max_x, top_level.max_y))

    def execute(self, statement, inputs=None):
        """Execute a prepared SQL statement on this geopackage database."""
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            if inputs is not None:
                result_cursor = cursor.execute(statement, inputs)
            else:
                result_cursor = cursor.execute(statement)
            return result_cursor

    def assimilate(self, source):
        """Assimilate .gpkg.part tiles into this geopackage database."""
        if not exists(source):
            raise IOError
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            cursor.execute("pragma synchronous = off;")
            cursor.execute("pragma journal_mode = off;")
            cursor.execute("pragma page_size = 65536;")
            #print "Merging", source, "into", self.__file_path, "..."
            query = "attach '" + source + "' as source;"
            cursor.execute(query)
            try:
                #
                # fix this pls
                #
                cursor.execute("""INSERT OR REPLACE INTO tiles
                (zoom_level, tile_column, tile_row, tile_data)
                SELECT zoom_level, tile_column, tile_row, tile_data
                FROM source.tiles;""")
                cursor.execute("detach source;")
            except Error as err:
                print("Error: {}".format(type(err)))
                print("Error msg:".format(err))
                raise
            remove(source)

    def __exit__(self, type, value, traceback):
        """Resource cleanup on destruction."""
        self.__db_con.close()


class TempDB(object):
    """
    Returns a temporary sqlite database to hold tiles for async workers.
    Has a <filename>.gpkg.part file format.
    """

    def __enter__(self):
        """With-statement caller."""
        return self

    def __init__(self, filename):
        """
        Constructor.

        Inputs:
        filename -- the filename this database will be created with
        """
        uid = uuid4()
        self.name = uid.hex + '.gpkg.part'
        self.__file_path = join(filename, self.name)
        self.__db_con = connect(self.__file_path)
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            stmt = """
                CREATE TABLE tiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zoom_level INTEGER NOT NULL,
                tile_column INTEGER NOT NULL,
                tile_row INTEGER NOT NULL,
                tile_data BLOB NOT NULL,
                UNIQUE (zoom_level, tile_column, tile_row));
            """
            cursor.execute(stmt)
            # Enable pragma for fast sqlite creation
            cursor.execute("pragma synchronous = off;")
            cursor.execute("pragma journal_mode = off;")
            cursor.execute("pragma page_size = 80000;")
            cursor.execute("pragma foreign_keys = 1;")
        self.image_blob_stmt = """
            INSERT INTO tiles
                (zoom_level, tile_column, tile_row, tile_data)
                VALUES (?,?,?,?)
        """

    def execute(self, statement, inputs=None):
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            if inputs is not None:
                result_cursor = cursor.execute(statement, inputs)
            else:
                result_cursor = cursor.execute(statement)
            return result_cursor

    def insert_image_blob(self, z, x, y, data):
        """
        Inserts a binary data array containing an image into a sqlite3
        database.

        Inputs:
        z -- the zoom level of the binary data
        x -- the row number of the data
        y -- the column number of the data
        data -- the image data containing in a binary array
        """
        #fixed order column/row -> y,x RB
        with self.__db_con as db_con:
            cursor = db_con.cursor()
            cursor.execute(self.image_blob_stmt, (z, y, x, data))

    def __exit__(self, type, value, traceback):
        """Resource cleanup on destruction."""
        self.__db_con.close()


def img_to_buf(img, img_type, jpeg_quality=75):
    """
    Returns a buffer array with image binary data for the input image.
    This code is based on logic implemented in MapProxy to convert PNG
    images to JPEG then return the buffer.

    Inputs:
    img -- an image on the filesystem to be converted to binary
    img_type -- the MIME type of the image (JPG, PNG)
    """
    defaults = {}
    buf = ioBuffer()
    if img_type == 'jpeg':
        img.convert('RGB')
        # Hardcoding a default compression of 75% for JPEGs
        defaults['quality'] = jpeg_quality
    elif img_type == 'source':
        img_type = img.format
    img.save(buf, img_type, **defaults)
    buf.seek(0)
    return buf


def img_has_transparency(img):
    """
    Returns a 0 if the input image has no transparency, 1 if it has some,
    and -1 if the image is fully transparent. Tiles *should be a perfect
    square (e.g, 256x256), so it can be safe to assume the first dimension
    will match the second.  This will ensure compatibility with different
    tile sizes other than 256x256.  This code is based on logic implemented
    in MapProxy to check for images that have transparency.

    Inputs:
    img -- an Image object from the PIL library
    """
    size = img.size[0]
    if img.mode == 'P':
        # For paletted images
        if img.info.get('transparency', False):
            return True
        # Convert to RGBA to check alpha
        img = img.convert('RGBA')
    if img.mode == 'RGBA':
        # Returns the number of pixels in this image that are transparent
        # Assuming a tile size of 256, 65536 would be fully transparent
        transparent_pixels = img.histogram()[-size]
        if transparent_pixels == 0:
            # No transparency
            return 0
        elif 0 < transparent_pixels < (size*size):
            # Image has some transparency
            return 1
        else:
            # Image is fully transparent, and can be discarded
            return -1
        #return img.histogram()[-size]
    return False


def file_count(base_dir, max_level):
    """
    A function that finds all image tiles in a base directory.  The base
    directory should be arranged in TMS format, i.e. z/x/y.

    Inputs:
    base_dir -- the name of the TMS folder containing tiles.

    Returns:
    A list of dictionary objects containing the full file path and TMS
    coordinates of the image tile.
    """
    print("Calculating number of tiles, this could take a while...")
    file_list = []
    # Avoiding dots (functional references) will increase performance of
    #  the loop because they will not be reevaluated each iteration.
    for root, sub_folders, files in walk(base_dir):
        #check the level
        lv = -1
        if len(root) > len(base_dir):
            lvStr = root[len(base_dir)+1:]
            if lvStr.startswith("L"):
               lvStr = lvStr[1:]
            m = re.search("\d*|L\d*", lvStr)
            if m:
                lv = m.group()
        if not lv:
            continue
        if int(lv) <= max_level or max_level == -1:
            temp_list = [join(root, f) for f in files if f.endswith(IMAGE_TYPES)]
            file_list += temp_list
    print("Found {} total tiles (max level:{}).".format(len(file_list), max_level))
    return [split_all(item) for item in file_list]


def split_all(path):
    """
    Function that parses TMS coordinates from a full images file path.

    Inputs:
    path -- a full file path to an image tile.

    Returns:
    A dictionary containing the TMS coordinates of the tile and its full
    file path.
    """
    parts = []
    full_path = path
    is_esri = 0
    # Parse out the tms coordinates
    for i in xrange(3):
        head, tail = split(path)
        #handle Esri Cache (Row Column)
        if tail.startswith("R") or tail.startswith("C"):
            parts.append(tail[1:])
            is_esri = 1
        #handle Esri Cache Level
        elif tail.startswith("L"):
            parts.append(tail[1:])
            is_esri = 1
        #TMS, WMTS Cache
        else:
            parts.append(tail)
        path = head

    if not is_esri:
        file_dict = dict(y=int(parts[0].split('.')[0]), x=int(parts[1]),z=int(parts[2]), path=full_path)
    else:
        #handle Esri Cache
        file_dict = dict(y=int(parts[0].split('.')[0], 16), x=int(parts[1], 16),z=int(parts[2]), path=full_path)
    return file_dict


def worker_map(temp_db, tile_dict, extra_args, invert_y):
    """
    Function responsible for sending the correct oriented tile data to a
    temporary sqlite3 database.

    Inputs:
    temp_db -- a temporary sqlite3 database that will hold this worker's tiles
    tile_dict -- a dictionary with TMS coordinates and file path for a tile
    tile_info -- a list of ZoomMetadata objects pre-generated for this tile set
    imagery -- the type of image format to send to the sqlite3 database
    invert_y -- a function that will flip the Y axis of the tile if present
    """
    tile_info = extra_args['tile_info']
    imagery = extra_args['imagery']
    jpeg_quality = extra_args['jpeg_quality']
    zoom = tile_dict['z']
    level = next((item for item in tile_info if item.zoom == zoom), None)
    x_row = tile_dict['x'] - level.min_tile_row
    if invert_y is not None:
        y_offset = invert_y(tile_dict['z'], level.max_tile_col)
        y_column = invert_y(tile_dict['z'], tile_dict['y'])
        y_column -= y_offset
    else:
        y_column = tile_dict['y'] - level.min_tile_col
    if IOPEN is not None:
        img = IOPEN(tile_dict['path'], 'r')
        data = ioBuffer()
        if imagery == 'mixed':
            if img_has_transparency(img):
                data = img_to_buf(img, 'png', jpeg_quality).read()
            else:
                data = img_to_buf(img, 'jpeg', jpeg_quality).read()
        else:
            data = img_to_buf(img, imagery, jpeg_quality).read()
        temp_db.insert_image_blob(zoom, x_row, y_column, sbinary(data))
    else:
        file_handle = open(tile_dict['path'], 'rb')
        data = buffer(file_handle.read())
        temp_db.insert_image_blob(zoom, x_row, y_column, data)
        file_handle.close()


def sqlite_worker(file_list, extra_args):
    """
    Worker function called by asynchronous processes.  This function
    iterates through a set of tiles to process them into a TempDB object.

    Inputs:
    file_list -- an array containing a subset of tiles that will be processed
                 by this function into a TempDB object
    base_dir -- the directory in which the geopackage will be created,
                .gpkg.part files will be generated here
    metadata -- a ZoomLevelMetadata object containing information about
                the tiles in the TMS directory
    """
    temp_db = TempDB(extra_args['root_dir'])
    with TempDB(extra_args['root_dir']) as temp_db:
        invert_y = None
        if extra_args['lower_left']:
            if extra_args['srs'] == 3857:
                invert_y = Mercator.invert_y
            elif extra_args['srs'] == 4326:
                invert_y = Geodetic.invert_y
            elif extra_args['srs'] == 3395:
                invert_y = EllipsoidalMercator.invert_y
            elif extra_args['srs'] == 9804:
                invert_y = ScaledWorldMercator.invert_y
            elif extra_args['srs'] == 21781:
                invert_y = CH1903LV03.invert_y
            elif extra_args['srs'] == 2056:
                invert_y = CH1903pLV95.invert_y

        [worker_map(temp_db, item, extra_args, invert_y) for item in file_list]


def allocate(cores, pool, file_list, extra_args):
    """
    Recursive function that fairly distributes tiles to asynchronous worker
    processes.  For N processes and C cores, N=C if C is divisible by 2.  If
    not, then N is the largest factor of 8 that is still less than C.
    """
    if cores is 1:
        print("Spawning worker with {} files".format(len(file_list)))
        return [pool.apply_async(sqlite_worker, [file_list, extra_args])]
    else:
        files = len(file_list)
        head = allocate(int(cores/2), pool, file_list[:int(files/2)], extra_args)
        tail = allocate(int(cores/2), pool, file_list[int(files/2):], extra_args)
        return head + tail


def build_lut(file_list, lower_left, srs, max_level, tile_matrix):
    """
    Build a lookup table that aids in metadata generation.

    Inputs:
    file_list -- the file_list dict made with file_count()
    lower_left -- bool indicating tile grid numbering scheme (tms or wmts)
    srs -- the spatial reference system of the tile grid

    Returns:
    An array of ZoomLevelMetadata objects that describe each zoom level of the
    tile grid.
    """
    # Initialize a projection class
    if srs == 3857:
        projection = Mercator()
    elif srs == 4326:
        projection = Geodetic()
    elif srs == 9804:
        projection = ScaledWorldMercator()
    elif srs == 21781:
        projection = CH1903LV03()
    elif srs == 2056:
        projection = CH1903pLV95()
    else:
        projection = EllipsoidalMercator()

    #setup tile matrix if specified
    if tile_matrix == "swiss_lv03":
        projection.setTileMatrix(SwissTM_LV03())
    elif tile_matrix == "swiss_lv95":
        projection.setTileMatrix(SwissTM_LV95())
    elif tile_matrix == "swiss_esri_lv03":
        projection.setTileMatrix(SwissTM_ESRI_LV03())

    # Create a list of zoom levels from the base directory
    zoom_levels = list(set([int(item['z']) for item in file_list]))
    zoom_levels.sort()
    matrix = []
    # For every zoom in the list...
    for zoom in zoom_levels:
        # create a new ZoomMetadata object...
        level = ZoomMetadata()
        level.zoom = zoom
        # Sometimes, tiling programs do not generate the folders responsible
        # for the X axis if no tiles are being made within them.  This results
        # in tiles "shifting" because of the way they are renumbered when
        # placed into a geopackage.
        # To fix, is there a zoom level preceding this one...
        if zoom - 1 in [item for item in zoom_levels if item == (zoom - 1)]:
            # Calculate the width and height
            try:
                level.matrix_width = projection.tile_matrix.scales[level.zoom][1][0]
                level.matrix_height = projection.tile_matrix.scales[level.zoom][1][1]
                level.min_tile_row = 0
                level.min_tile_col = 0
                level.max_tile_row = level.matrix_height
                level.max_tile_col = level.matrix_width
            except:
                # there is, now retrieve it....
                (prev,) = ([item for item in matrix if item.zoom == (zoom - 1)])
                # and fix the grid alignment values
                level.min_tile_row = 2 * prev.min_tile_row
                level.min_tile_col = 2 * prev.min_tile_col
                level.max_tile_row = 2 * prev.max_tile_row + 1
                level.max_tile_col = 2 * prev.max_tile_col + 1
                level.matrix_width = prev.matrix_width * 2
                level.matrix_height = prev.matrix_height * 2
        else:
            # Get all possible x and y values...
            x_vals = [int(item['x']) for item in file_list
                      if int(item['z']) == zoom]
            y_vals = [int(item['y']) for item in file_list
                      if int(item['z']) == zoom]
            # then get the min/max values for each.
            level.min_tile_row, level.max_tile_row = min(x_vals), max(x_vals)
            level.min_tile_col, level.max_tile_col = min(y_vals), max(y_vals)

            # Fill in the matrix width and height for this top level
            try:
                level.matrix_width = projection.tile_matrix.scales[level.zoom][1][0]
                level.matrix_height = projection.tile_matrix.scales[level.zoom][1][1]
                level.min_tile_row = 0
                level.min_tile_col = 0
                level.max_tile_row = level.matrix_height - 1
                level.max_tile_col = level.matrix_width - 1
            except:
                x_width_max = max([item['x'] for item in file_list
                                   if item['z'] == level.zoom])
                x_width_min = min([item['x'] for item in file_list
                                   if item['z'] == level.zoom])
                level.matrix_width = (x_width_max - x_width_min) + 1
                y_height_max = max([item['y'] for item in file_list
                                   if item['z'] == level.zoom])
                y_height_min = min([item['y'] for item in file_list
                                   if item['z'] == level.zoom])
                level.matrix_height = (y_height_max - y_height_min) + 1
        if lower_left:
            # TMS-style tile grid, so to calc the top left corner of the grid,
            # you must get the min x (row) value and the max y (col) value + 1.
            # You are adding 1 to the y value because the math to calc the
            # coord assumes you want the bottom left corner, not the top left.
            # Similarly, to get the bottom right corner, add 1 to x value.
            #
            #ORIGINAL
            #
##            level.min_x, level.max_y = projection.get_coord(
##                level.zoom, level.min_tile_row, level.max_tile_col + 1)
##            level.max_x, level.min_y = projection.get_coord(
##                level.zoom, level.max_tile_row + 1, level.min_tile_col)

            #
            # Modified by bam: column shd come first, then row: lat,long
            #
            level.min_x, level.max_y = projection.get_coord(
                level.zoom, level.max_tile_col + 1, level.min_tile_row)
            level.max_x, level.min_y = projection.get_coord(
                level.zoom, level.min_tile_col, level.max_tile_row + 1)

        else:
            # WMTS-style tile grid, so to calc the top left corner of the grid,
            # you must get the min x (row value and the min y (col) value + 1.
            # You are adding 1 to the y value because the math to calc the
            # coord assumes you want the bottom left corner, not the top left.
            # Similarly, to get the bottom right corner, add 1 to x value.
            # -- Since this is WMTS, we must invert the Y axis before we calc

            # its not more required I guess
            #inv_min_y = projection.invert_y(level.zoom, level.min_tile_col)
            #inv_max_y = projection.invert_y(level.zoom, level.max_tile_col)

            #
            #ORIGINAL
            #

            #inv_min_y = projection.invert_y(level.zoom, level.min_tile_col)
            #inv_max_y = projection.invert_y(level.zoom, level.max_tile_col)

            #level.min_x, level.max_y = projection.get_coord(
            #    level.zoom, level.min_tile_row, inv_min_y + 1)
            #level.max_x, level.min_y = projection.get_coord(
            #    level.zoom, level.max_tile_row + 1, inv_max_y)

            level.min_x, level.max_y = projection.get_coord(
                level.zoom, level.min_tile_col, level.min_tile_row - 1)
            level.max_x, level.min_y = projection.get_coord(
                level.zoom, level.max_tile_col + 1, level.max_tile_row)
        # Finally, add this ZoomMetadata object to the list
        matrix.append(level)
    return matrix


def combine_worker_dbs(out_geopackage):
    """
    Searches for .gpkg.part files in the base directory and merges them
    into one Geopackage file

    Inputs:
    out_geopackage -- the final output geopackage file
    """
    base_dir = split(out_geopackage.file_path)[0]
    if base_dir == "":
        base_dir = "."
    glob_path = join(base_dir + '/*.gpkg.part')
    file_list = glob(glob_path)
    print("Merging temporary databases...")
    #[out_geopackage.assimilate(f) for f in file_list]
    itr = len(file_list)
    status = ["|", "/", "-", "\\"]
    counter = 0
    for tdb in file_list:
        comp = len(file_list) - itr
        itr -= 1
        out_geopackage.assimilate(tdb)
        if tdb == file_list[-1]:
            stdout.write("\r[X] Progress: [" + "=="*comp + "  "*itr + "]")
        else:
            stdout.write("\r[" + status[counter] + "] Progress: [" + "=="*comp + "  "*itr + "]")
        stdout.flush()
        if counter != len(status)-1:
            counter += 1
        else:
            counter = 0
    print(" All geopackages merged!")


def main(arg_list):
    """
    Create a geopackage from a directory of tiles arranged in TMS or WMTS
    format.

    Inputs:
    arg_list -- an ArgumentParser object containing command-line options and
    flags
    """
    #Is a max level specified
    max_level = arg_list.maxlvl

    # Build the file dictionary
    files = file_count(arg_list.source_folder, max_level)
    if len(files) == 0:
        # If there are no files, exit the script
        print(" Ensure the correct source tile directory was specified.")
        exit(1)

    # Is a tilematrix specified?
    tile_matrix=arg_list.tm

    # Is the input tile grid aligned to lower-left or not?
    lower_left = arg_list.tileorigin == 'll' or arg_list.tileorigin == 'sw'
    # Get the output file destination directory
    root_dir, _ = split(arg_list.output_file)
    # Build the tile matrix info object
    tile_info = build_lut(files, lower_left, arg_list.srs, max_level, tile_matrix)
    # Initialize the output file
    if arg_list.threading:
        # Enable tiling on multiple CPU cores
        cores = cpu_count()
        pool = Pool(cores)
        # Build allocate dictionary
        extra_args = dict(root_dir=root_dir, tile_info=tile_info,
                lower_left=lower_left, srs=arg_list.srs,
                imagery=arg_list.imagery, jpeg_quality=arg_list.q)
        results = allocate(cores, pool, files, extra_args)
        status = ["|", "/", "-", "\\"]
        counter = 0
        try:
            while True:
                rem = sum([1 for item in results if not item.ready()])
                if rem == 0:
                    stdout.write("\r[X] Progress: [" + "=="*(cores-rem) +
                            "  "*rem + "]")
                    stdout.flush()
                    print(" All Done!")
                    break
                else:
                    stdout.write("\r[" + status[counter] + "] Progress: [" +
                            "=="*(cores-rem) + "  "*rem + "]")
                    stdout.flush()
                    if counter != len(status)-1:
                        counter += 1
                    else:
                        counter = 0
                sleep(.25)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print(" Interrupted!")
            pool.terminate()
            exit(1)
    else:
        # Debugging call to bypass multiprocessing (-T)
        extra_args = dict(root_dir=root_dir, tile_info=tile_info,
                lower_left=lower_left, srs=arg_list.srs,
                imagery=arg_list.imagery, jpeg_quality=arg_list.q)
        sqlite_worker(files, extra_args)
    # Combine the individual temp databases into the output file
    with Geopackage(arg_list.output_file, arg_list.srs, tile_matrix) as gpkg:
        combine_worker_dbs(gpkg)
        # Using the data in the output file, create the metadata for it
        gpkg.update_metadata(tile_info)
    print("Complete")

if __name__ == '__main__':
    print("""
        tiles2gpkg_parallel.py  Copyright (C) 2014  Reinventing Geospatial, Inc
        This program comes with ABSOLUTELY NO WARRANTY.
        This is free software, and you are welcome to redistribute it
        under certain conditions.
    """)
    PARSER = ArgumentParser(description="Convert TMS folder into geopackage")
    PARSER.add_argument("source_folder", metavar="source",
            help="Source folder of TMS files.")
    PARSER.add_argument("output_file", metavar="dest",
            help="Destination file path.")
    PARSER.add_argument("-tileorigin", metavar="tile_origin",
            help="Tile point of origin location. Valid options " +
            "are ll, ul, nw, or sw.", choices=["ll", "ul", "sw", "nw"],
            default="ll")
    PARSER.add_argument("-maxlvl", metavar="max_level", type=int, default=-1,
            help="Maximum cache level to package (0 based index), 0-100. Default is -1 = all",
            choices=list(range(100)))
    PARSER.add_argument("-tm", metavar="tile_matrix", default="",
            help="Tilematrix name to use. Default is the regular one. Choices are swiss_lv03, swiss_lv95, swiss_esri_lv03",
            choices=["swiss_lv03", "swiss_lv95", "swiss_esri_lv03"])
    PARSER.add_argument("-srs", metavar="srs", help="Spatial reference " +
            "system. Valid options are 3857, 4326, 3395, and 9804.",
            type=int, choices=[3857, 4326, 3395, 9804, 21781, 2056], default=3857)
    PARSER.add_argument("-imagery", metavar="imagery",
            help="Imagery type. Valid options are mixed, " +
            "jpeg, png, or source.", choices=["mixed", "jpeg", "png", "source"],
            default="source")
    PARSER.add_argument("-q", metavar="quality", type=int, default=75,
            help="Quality for jpeg images, 0-100. Default is 75",
            choices=list(range(100)))
    PARSER.add_argument("-a", dest="append", action="store_true",
            default=False, help="Append tile set to existing geopackage")
    PARSER.add_argument("-T", dest="threading", action="store_false",
            default=True, help="Disable multiprocessing.")
    ARG_LIST = PARSER.parse_args()
    if not exists(ARG_LIST.source_folder) or exists(ARG_LIST.output_file):
        PARSER.print_usage()
        print("Ensure that TMS directory exists and out file does not.")
        exit(1)
    if ARG_LIST.q is not None and ARG_LIST.imagery == 'png':
        PARSER.print_usage()
        print("-q cannot be used with png")
        exit(1)
    main(ARG_LIST)
