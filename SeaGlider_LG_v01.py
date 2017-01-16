#!/usr/bin/env python
"""
UAGE:   from SeaGlider_LG import SeaGlider
        sg = SeaGlider(/path/to/folder/)

For more info see help(SeaGlider)
"""

import re
import os
from glob import glob
from netCDF4 import Dataset
from numpy import array, arange, unique, datetime64, ma, where, nan, ndarray, \
    interp, sort
from pandas import DataFrame, Series


class SeaGlider(object):
    __doc__ = """
    This is a class that reads in glider data from seaglider *.nc files. This
    is designed to be used interactively.

    USAGE:  sg = SeaGlider(/path/to/folder/)
            sg = SeaGlider(/path/to/folder/*wildcard*.nc)

    The output object contains dynamic methods for each plottable variable,
    which are a subclass SG_var:
            sg.temperature
            sg.salinity
    SG_var subclass is interpolated when data is read in, or each profile can
    be imported individually by calling sg_var[i]
    """

    def __init__(self, data_directoy, ref_file_name=None):

        self._process_directory_name(data_directoy)

        if ref_file_name is None:
            ref_file_name = self.files[0]
        self.reference_file = ref_file_name

        nc0 = Dataset(self.files[0])
        nc1 = Dataset(self.files[-1])
        ncr = Dataset(self.reference_file)
        self.variables = {}
        ref_size = ncr.variables['depth'].size

        for key in ncr.variables:
            if ref_size == ncr[key].size:
                var_obj = SG_var(self.files, key)
                self.variables[key] = var_obj
                setattr(self, key, self.variables[key])
            elif ncr[key].dtype == 'float':
                setattr(self, key + '_placeholder', object)

        t0 = datetime64(nc0.getncattr('time_coverage_start').replace('Z', ''))
        t1 = datetime64(nc1.getncattr('time_coverage_end').replace('Z', ''))
        self.date_range = array([t0, t1], dtype='datetime64[s]')

    def __getitem__(self, key):
        return getattr(self, key)

    def _process_directory_name(self, data_directoy):
        split = os.path.split
        data_directoy = re.sub('/$', '/*.nc', data_directoy)
        self.directory = data_directoy
        self.files = sort(glob(data_directoy))
        self.dives = array([int(split(f)[-1][4:-3]) for f in self.files])

        glider = array([(split(f)[-1][:4]) for f in self.files])
        glider = unique(glider)
        assert glider.size == 1, 'More than one glider included in directory'
        self.glider = glider[0]

        return

    def __repr__(self):

        string = ""
        string += "Number of files:     {}\n".format(self.files.size)
        string += "Date range:          {} to {}".format(
            self.date_range[0], self.date_range[1]).replace('T', ' ')
        string += "\n"
        plot_vars = sort(self.variables.keys()).astype(str)
        string += "Plot variables: \n{}".format(plot_vars)
        string += '\n\n'

        return string

    def __str__(self):
        return self.__repr__()


class SG_var(object):
    from matplotlib import cm as _cm

    def __init__(self, files, name):
        from numpy import array
        self.name = name
        self.__files__ = array(files)
        self.__dataframe__ = []
        self.__values__ = []
        self.attrs = {}

        # Date range
        nc0 = Dataset(files[0])
        nc1 = Dataset(files[-1])
        t0 = datetime64(nc0.getncattr('time_coverage_start').replace('Z', ''))
        t1 = datetime64(nc1.getncattr('time_coverage_end').replace('Z', ''))
        self.date_range = array([t0, t1], dtype='datetime64[s]')

        for key in nc0.variables[name].ncattrs():
            self.attrs[key] = nc0.variables[name].getncattr(key)

    def __getitem__(self, key):
        from netCDF4 import Dataset
        from pandas import Series

        fname = self.__files__[key]
        print 'Importing file: {}'.format(fname)
        nco = Dataset(fname)
        depth = nco.variables['depth'][:]
        var = nco.variables[self.name]
        data = Series(var[:], index=depth, name=self.name)

        return data

    def _load(self):
        dataframe = self._read_glider_var(self.__files__, self.name)
        self.__dataframe__ = dataframe.astype(float)
        self.__values__ = ma.masked_invalid(self.__dataframe__.values)

    @property
    def data(self):
        if not any(self.__dataframe__):
            self._load()
        return self.__dataframe__

    @property
    def values(self):
        if not any(self.__dataframe__):
            self._load()
        return self.__values__

    @property
    def ix(self):
        return self.data.ix

    def _read_netcdf_var(self, fname, key):
        nc = Dataset(fname)

        depth = nc.variables['depth'][:]
        raw = ma.masked_invalid(nc.variables[key][:])
        mask = ~raw.mask
        depth = depth[mask]
        raw = raw[mask]
        dn = where(arange(depth.size) < depth.argmax())[0]
        up = where(arange(depth.size) >= depth.argmax())[0][::-1]

        xi = arange(1000.)
        intp = ndarray([1000, 2])
        intp[:, 1] = interp(xi, depth[dn], raw[dn], right=nan, left=nan)
        intp[:, 0] = interp(xi, depth[up], raw[up], right=nan, left=nan)

        return intp

    def _read_glider_var(self, files, key):

        depth = Series(arange(1000), name='depth')
        dives = Series(arange(0, files.size, 0.5) + 1, name='dives')
        data = DataFrame([], index=depth, columns=dives)

        for c, fname in enumerate(files):
            i = c * 2, c * 2 + 1
            try:
                data.iloc[:, i] = self._read_netcdf_var(fname, key)
            except (KeyError, ValueError):
                data.iloc[:, i] = ndarray([1000, 2]) * nan
                print "{} doesn't contain {} or has no data".format(fname, key)

        return data

    def __repr__(self):
        string = ""
        string += "Variable:         {}\n".format(self.name)
        string += "Number of dives:  {}\n".format(self.__files__.size)
        string += "Date range:       {} to {}".format(
            self.date_range[0], self.date_range[1]).replace('T', ' ')
        string += "\n"
        for key in self.attrs:
            string += '{0: <18}{1}\n'.format(key.capitalize(), self.attrs[key])
        string += '\n\n'
        if any(self.__dataframe__):
            string += "Data has been interpolated linearly to nearest metre\n"
            string += "Dive index on the integer are down, on the .5 is up\n"
            string += self.data.head().ix[:, :3.5].__repr__()
        else:
            string += 'Data not loaded yet [.data .values]'

        return string

    def __str__(self):
        string = self.__repr__()
        return string

    def plot_section(self, cmap=_cm.Spectral_r, vlim=None):
        import matplotlib.pyplot as plt
        from numpy import arange, percentile
        z = self.values
        y = arange(z.shape[0])
        x = arange(z.shape[1])
        if not vlim:
            vmin, vmax = percentile(z[~z.mask], [0.5, 99.5])
        else:
            vmin, vmax = vlim
        dict.has_key
        vname = self.name.capitalize()
        units = '({})'.format(
            self.attrs['units']) if 'units' in self.attrs else ''
        clabel = '{} {}'.format(vname, units)

        fig, ax = plt.subplots(1, 1, figsize=[11, 4])
        im = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label(clabel, rotation=-90, va='bottom')
        ax.set_xlim(0, x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Profile number')

        return ax


if __name__ == '__main__':
    data_dir = '/Users/luke/Desktop/SeaGlider/sg_data/'
    sg = SeaGlider(data_dir + 'p5420*')
    print sg
