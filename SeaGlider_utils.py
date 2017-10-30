#!/usr/bin/env python
"""
UAGE:   from SeaGlider_LG import SeaGlider
        sg = SeaGlider(/path/to/folder/)

For more info see help(SeaGlider)

written by Luke Gregor (lukegre@gmail.com)


Wishlist
--------
- SeaGlider must read variables where N != len(depth)
- SeaGlider must be able to load two variables at once
  to save time, rather than loop through everything
"""

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SeaGlider:
    __doc__ = """
    This is a class that reads in glider data from seaglider *.nc files. This
    is designed to be used interactively.

    USAGE:  sg = SeaGlider(/path/to/folder/)
            sg = SeaGlider(/path/to/folder/*wildcard*.nc)

    The output object contains dynamic methods for each plottable variable,
    which are a subclass SeaGlider_variable:
            sg.temperature
            sg.salinity
    SeaGlider_variable subclass is interpolated when data is read in, or each
    profile can be imported individually by calling SeaGlider_variable[i]
    """

    def __init__(self, data_directoy, preload_variables=None, ref_file_name=None):
        from numpy import array, datetime64
        from netCDF4 import Dataset

        self._process_directory_name(data_directoy)

        if ref_file_name is None:
            ref_file_name = self.files[0]
        self.reference_file = ref_file_name

        nc0 = Dataset(self.files[0])
        nc1 = Dataset(self.files[-1])
        ncr = Dataset(self.reference_file)

        self.variables = {}
        self.coords = {
            'depth': 'ctd_depth',
            'time': 'ctd_time',
            'lat': 'latitude',
            'lon': 'longitude'
        }

        ref_size = ncr.variables[self.coords['time']].size

        for key in ncr.variables:
            if ref_size == ncr[key].size:
                var_obj = SeaGlider_variable(self.files, key, self.coords)
                self.variables[key] = var_obj
                setattr(self, key, self.variables[key])
            elif ncr[key].dtype == 'float':
                setattr(self, key + '_placeholder', object)


        t0 = datetime64(nc0.getncattr('time_coverage_start').replace('Z', ''))
        t1 = datetime64(nc1.getncattr('time_coverage_end').replace('Z', ''))
        self.date_range = array([t0, t1], dtype='datetime64[s]')

        nc0.close()
        nc1.close()
        ncr.close()

        if preload_variables:
            if type(preload_variables) is str:
                preload_variables = [preload_variables]
            self[preload_variables]


    def __getitem__(self, key):
        from xarray import open_dataset

        if type(key) == int:
            fname = self.files[key]
            nco = open_dataset(fname)
            return nco
        elif type(key) == list:
            if all([type(k) is str for k in key]):
                self.load_multiple_vars(key)
        else:
            return getattr(self, key)

    def _process_directory_name(self, data_directoy):
        from numpy import sort, array, unique
        from glob import glob
        import re
        import os

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
        string += "Coordinates:         "
        string += str(self.coords)
        string += "\n"
        plot_vars = len(self.variables.keys())
        string += "Number of plotting variables: {} [press TAB to autocomplete]".format(plot_vars)
        string += '\n\n'

        return string

    def __str__(self):
        return self.__repr__()

    def load_multiple_vars(self, keys):

        df = SeaGlider_variable._read_glider_var(SeaGlider,
            self.files,
            keys + list(self.coords.values()),
            depth=self.coords['depth'],
            time=self.coords['time'],
        )

        for key in keys:
            dims = list(self.coords.values())
            cols = [key] + dims + ['dive']
            sg_var = getattr(self, key)
            setattr(sg_var, '__dataframe__', df[cols])
            setattr(sg_var, '__values__', df[cols].values)


class SeaGlider_variable:
    """
    Variable metadata is loaded initally.
    Data is only loaded when required - this only has to be done once.
    Thereafter, the data is stored in the object.

    Note that default time, depth, latitude and longitude names for the
    variables are set. These can be changed in the config of `SeaGlider`.
    """

    def __init__(self, files, name, coords):
        from numpy import array, datetime64, r_ as concat_by_row, arange
        from netCDF4 import Dataset

        self.name = name
        self.__files__ = array(files)
        self.__dataframe__ = []
        self.__gridded__ = []
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

        nc0.close()
        nc1.close()

        for dim in coords:
            setattr(self, dim + "_name", coords[dim])

        self.depth_bins = concat_by_row[
            arange(0, 100, 0.5),
            arange(100, 400, 1.),
            arange(400, 1000, 2.)
        ]

    def _load(self):

        keys = self.name, self.time_name, self.depth_name, self.lat_name, self.lon_name

        dataframe = self._read_glider_var(
            self.__files__, keys,
            depth=self.depth_name,
            time=self.time_name)

        self.__dataframe__ = dataframe
        self.__values__ = self.__dataframe__.values

    @property
    def gridded(self):
        """
        Object where the gridded/binned data is stored.
        If binning has not been performed default settings will be used.
        See `.bin_depths` for the details. Note that `.bin_depths` can
        also be used to overwrite the stored gridded data.
        """
        if not any(self.__gridded__):
            dataframe = self.bin_depths()
            self.__gridded__ = dataframe
        return self.__gridded__

    @property
    def data(self):
        """
        Object where data is stored as a pandas.DataFrame.
        If data has not been loaded, this object will do so dynamically.
        """
        if not any(self.__dataframe__):
            self._load()
        return self.__dataframe__

    @property
    def loc(self):
        """
        Access the DataFrame directly rather than having to go `.data.loc`.
        """
        return self.data.loc

    def _read_glider_var(self, files, keys, depth='ctd_depth', time='ctd_time'):
        from pandas import concat, DataFrame
        from tqdm import tnrange, tqdm_notebook
        from xarray.conventions import decode_cf_datetime
        from netCDF4 import Dataset

        data = []
        print("Loading: {}".format(str(keys)[1:-1].replace("'", "")))
        for i in tnrange(files.size):
            fname = files[i]
            try:
                # READ IN NETCDF
                nc = Dataset(fname)
                n = nc.variables[time].size
                df = DataFrame([], columns=keys, index=range(n))
                for key in keys:
                    df[key] = nc.variables[key][:]
                nc.close()

                # ASSIGNING DIVE INDEX
                df['dive'] = i
                # up dives are `dive_number + 0.5`
                max_depth = df[depth].argmax()
                df.loc[max_depth:, 'dive'] += 0.5

                # APPEND DATA TO LIST
                data += df,
            except (KeyError, ValueError):
                print("{} doesn't contain {} / has no data".format(fname, key))
        data = concat(data, ignore_index=True)

        # fixing times
        nct = Dataset(fname)
        time_unit = nct.variables[time].getncattr('units')
        data[time] = decode_cf_datetime(data[time], time_unit)
        data = data.set_index(time, drop=False)
        nct.close()

        return data

    def __repr__(self):
        string = ""
        string += "Variable:         {}\n".format(self.name)
        string += "Number of dives:  {}\n".format(self.__files__.size)
        string += "Date range:       {} to {}\n".format(
            self.date_range[0], self.date_range[1]).replace('T', ' ')
        string += "\n"
        for key in self.attrs:
            string += '{0: <18}{1}\n'.format(key.capitalize(), self.attrs[key])
        string += "\nDefault coordinate names [Change these as needed]\n"
        string += "     depth_name   '{}'\n".format(self.depth_name)
        string += "     time_name    '{}'\n".format(self.time_name)
        string += "     lat_name     '{}'\n".format(self.lat_name)
        string += "     lon_name     '{}'\n".format(self.lon_name)

        string += '\n\n'
        if any(self.__dataframe__):
            string += "Data is not interpolated/binned [.binned].\n"
            string += self.data.head().__repr__()
        else:
            string += 'Data not loaded yet [.data .values]'

        return string

    def __str__(self):
        string = self.__repr__()
        return string

    def bin_depths(self, bins=None, how='mean'):
        """
        This function bins the variable to set depths.
        If no depth is specified it defaults to:
            start : step : end
            ------------------
                0 :  0.5 : 100
              100 :  1.0 : 400
              400 :  2.0 : max
            ------------------
        This is the sampling method typically used by the CSIR.
        This can be easily changed by specifying an array of increasing values.

        The function used to bin the data can also be set, where the default
        is the mean. This can be changed to 'median', 'std', 'count', etc...

        Lastly, the data is stored as SeaGliderVariable.gridded for future
        easy access. If you would like to regrid the data to another grid
        use SeaGliderVariable.bin_depths().
        """

        from pandas import cut
        from pylab import date2num
        from scipy.stats import mode
        from numpy import r_ as concat_by_row, arange, diff

        depth = self.depth_name
        if bins is None:
            # this is the CSIR default that is chosen to save battery
            bins = self.depth_bins
        else:
            self.depth_bins = bins

        if bins.max() < self.data[depth].max():
            print ('extending preset bins to max depth')
            step = mode(diff(bins[-10:])).mode
            start = bins[-1] + step
            stop = self.data[depth].max() + step
            extended_bins = arange(start, stop, step)
            bins = concat_by_row[bins, extended_bins]

        labels = (bins[:-1] + bins[1:]) / 2.
        bins = cut(self.data[depth], bins, labels=labels)
        dive = self.data.dive

        grp = self.data.groupby([dive, bins])
        grouped = getattr(grp[self.name], how)()
        gridded = grouped.unstack(level=0)

        grp_cols = self.data[self.name].groupby(dive)
        gridded.columns = [grp_cols.groups[k][0] for k in grp_cols.groups]
        self.__gridded__ = gridded

        return gridded

    def scatter(self, **kwargs):
        """
        Plot a scatter plot of the dives with x-time and y-depth and
        c-variable.
        The **kwargs can be anything that gets passed to plt.scatter.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from numpy import ma
        from matplotlib.pyplot import scatter

        x = self.data.index.to_pydatetime()
        y = self.data[self.depth_name]
        z = ma.masked_invalid(self.data[self.name])
        # ma.masked_invalid to be consistent with pcolormesh
        s = ma.masked_invalid(kwargs['s'] if 's' in kwargs else 20)
        ax = self._plot_section(scatter, x, y, s, c=z, **kwargs)

        return ax

    def pcolormesh(self, interpolated=True, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable.
        The **kwargs can be anything that gets passed to plt.pcolormesh.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from matplotlib.pyplot import pcolormesh
        from numpy import ma

        if interpolated:
            gridded = self.gridded.interpolate(method='linear', limit=4)
        else:
            gridded = self.gridded

        z = ma.masked_invalid(gridded)
        y = gridded.index.values
        x = gridded.columns.values

        ax = self._plot_section(pcolormesh, x, y, z, **kwargs)

        return ax

    def _plot_section(self, plot_func, x, y, z, **kwargs):
        from numpy import percentile
        from matplotlib.pyplot import subplots, colorbar

        if "vmin" not in kwargs:
            kwargs['vmin'] = percentile(z[~z.mask], 1)
        if "vmax" not in kwargs:
            kwargs['vmax'] = percentile(z[~z.mask], 99)

        vname = self.name.capitalize()
        units = '({})'.format(
            self.attrs['units']) if 'units' in self.attrs else ''
        clabel = '{} {}'.format(vname, units)

        fig, ax = subplots(1, 1, figsize=[11, 4])

        im = plot_func(x, y, z, **kwargs)
        cb = colorbar(mappable=im, pad=0.02, ax=ax)
        cb.set_label(clabel, rotation=-90, va='bottom')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()
        return ax

    def plot_depth_binning(self, bins=None):
        from numpy import diff, abs, linspace
        from matplotlib.pyplot import subplots, colorbar, cm
        from matplotlib.colors import LogNorm

        depth = self.data[self.depth_name]
        x = abs(diff(depth))
        y = depth[1:]

        # plotting
        fig, ax = subplots(figsize=[3, 5])
        hbx = linspace(0, 4, 60)
        hby = linspace(0, y.max(), 100)
        im = ax.hist2d(x, y, bins=[hbx, hby], cmap=cm.Blues, norm=LogNorm())
        ax.set_ylim(y.max(), 0)
        ax.set_xlim(0, 4)
        ax.grid(True)

        ax.set_title('Histogram of sea glider \nvertical sampling resolution')
        ax.set_ylabel('Depth [m]')
        ax.set_xlabel('Depth difference (m)')

        cb = colorbar(im[-1])
        cb.set_label('Number of measurements', rotation=-90, va='bottom')

        if bins is None:
            bins = self.depth_bins
        else:
            self.depth_bins = bins

        xb = abs(diff(bins))
        yb = bins[1:]
        ax.plot(xb, yb, color='orange', lw=4)

        return ax


if __name__ == '__main__':
    data_dir = '/Users/luke/Documents/SeaGliders/Data/Level_0/sg543/2015_SAGE/'
    sg = SeaGlider(data_dir + 'p5430*')
    print(sg)
