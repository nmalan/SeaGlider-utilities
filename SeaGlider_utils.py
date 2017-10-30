#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UAGE:   from SeaGlider_utils import SeaGlider
        sg = SeaGlider('/path/to/folder/')

For more info see help(SeaGlider)

written by Luke Gregor (lukegre@gmail.com)
"""

from __future__ import print_function
from netCDF4 import Dataset
from pandas import DataFrame, Series
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SeaGlider:

    def __init__(self, expidition_directory, config_dict=None):
        from glob import glob
        dt64 = np.datetime64

        # setting up standard locs
        self.directory = expidition_directory
        self.files = np.sort(glob(self.directory))

        # loading data for dates and dummy vars
        nc0 = Dataset(self.files[0])
        nc1 = Dataset(self.files[-1])

        # creating dimensions where data is stored
        dims = np.array(list(nc0.dimensions.keys()))
        dims = dims[np.array([not d.startswith('string') for d in dims])]
        self.data = {}
        for key in dims:
            self.data[key] = DataFrame()
            self.data[key].var_count = 0
            self.data[key].dimension = key

        for key in nc0.variables:
            dim = nc0.variables[key].dimensions
            # dims can be a valid dimension, () or string_n
            if dim:  # catch empty tuples
                # there are dimensions that are just `string_n` placeholders
                if dim[0].startswith('string'):  # SeaGliderPointVariable
                    var_obj = _SeaGliderPointVariable(key, self.files)
                else:  # SeaGliderDiveVariable
                    dim_df = self.data[dim[0]]
                    dim_df.var_count += 1
                    var_obj = _SeaGliderDiveVariable(key, self.files, dim_df)
            else:  # SeaGliderPointVariable
                var_obj = _SeaGliderPointVariable(key, self.files)
            # assign as an object of SeaGlider
            setattr(self, key, var_obj)

        t0 = dt64(nc0.getncattr('time_coverage_start').replace('Z', ''))
        t1 = dt64(nc1.getncattr('time_coverage_end').replace('Z', ''))
        self.date_range = np.array([t0, t1], dtype='datetime64[s]')

        nc0.close()
        nc1.close()

    def __getitem__(self, key):
        from xarray import open_dataset

        if type(key) == int:
            fname = self.files[key]
            nco = open_dataset(fname)
            return nco
        elif type(key) == list:
            if all([type(k) is str for k in key]):
                self._load_multiple_vars(key)
            else:
                return "All arguements must be strings"
        elif type(key) == str:
            return getattr(self, key)
        else:
            return "Indexing with {} does not yet exist".format(key)

    def _load_multiple_vars(self, keys):

        # TO DO
        # This will load multple keys.
        # 1. find the dimensions for each of the keys and group accordingly
        # 2. iterate through groups and import with _read_nc_files
        # 3. assign each group to the appropriate dimension_dict
        pass

    def __repr__(self):
        dims = self.data
        dim_keys = list(dims.keys())

        string = ""
        string += "Number of dives:  {}\n".format(self.files.size)
        string += "Date range:       {}   to   {}".format(
            self.date_range[0], self.date_range[1]).replace('T', ' ')
        string += "\n"
        key = dim_keys[0]
        string += "Dimensions:       access these DataFrames in `.data[dim_name]`\n"
        for key in dim_keys:
            string += "                  • {} ({})\n".format(key, dims[key].var_count)
        string += "\n"
        string += "Access all variables directly from this object `.variable_name`\n"
        string += "[press TAB to autocomplete]"
        string += '\n\n'

        return string


class _SeaGliderDiveVariable:

    def __init__(self, name, files, dimension_df):
        nco = Dataset(files[0])

        self.__data__ = dimension_df
        self.__gridded__ = None
        self.__files__ = files
        self.name = name
        self.attrs = dict(nco.variables[name].__dict__)
        self.__keys__ = [name]
        if 'coordinates' in self.attrs:
            coordinates = self.attrs['coordinates'].split()
            self.__keys__ += coordinates
            self.__data__.coordinates = coordinates
        else:
            self.__data__.coordinates = []

        nco.close()

    @property
    def gridded(self):
        if self.__gridded__ is None:
            self.__gridded__ = self.bin_depths()
        return self.__gridded__

    @property
    def data(self):

        return self.load(return_data=True)

    def load(self, return_data=False):
        for k in self.__data__.coordinates:
            if k not in self.__keys__:
                self.__keys__ += k,
        keys = self.__keys__

        missing_keys = np.array([k not in self.__data__ for k in keys])

        # if statement will load the data only if columns are missing
        if any(missing_keys):
            missing_keys = np.array(keys)[missing_keys]
            df = self._read_nc_files(self.__files__, missing_keys)
            # if variable has coordinates and not yet loaded, will be processed
            if any([k in self.__data__.coordinates for k in missing_keys]):
                df = self._process_coords(df, self.__files__[0])
                if 'dives' in df:
                    self.__data__.coordinates += 'dives',
            # inplace assignment for the variables
            for col in df:
                self.__data__[col] = df[col]

        if ('dives' in self.__data__.coordinates) & ('dives' not in self.__keys__):
            self.__keys__ += 'dives',

        if return_data:
            return self.__data__.loc[:, self.__keys__]

    def __repr__(self):
        string = ""
        string += "Variable:         {}\n".format(self.name)
        string += "Number of dives:  {}\n".format(self.__files__.size)
        string += "Dimension:        {}\n".format(self.__data__.dimension)

        string += "Attributes:\n"
        for key in self.attrs:
            string += '    {0: <18}{1}\n'.format(key.capitalize() + ":", self.attrs[key])

        string += '\n'
        if self.name in self.__data__:
            string += "Data is not interpolated/binned \n"
            string += self.data.head().__repr__()
            string += "\n{} measurements".format(self.data.shape[0])
        else:
            string += 'Data not loaded yet [.data]'

        return string

    def _read_nc_files(self, files, keys, dives=True):
        from tqdm import trange

        data = []
        for i in trange(files.size):
            fname = files[i]
            nc = Dataset(fname)
            arr = np.r_[[nc.variables[k][:] for k in keys]]
            if dives:
                arr = np.r_[arr, np.ones([1, arr.shape[1]]) * i]
            nc.close()
            data += arr.T,

        data = np.concatenate(data)
        cols = list(keys) + (['dives'] if dives else [])
        df = DataFrame(data, columns=cols)

        return df

    def _process_coords(self, df, reference_file_name):

        # TRY TO GET DEPTH AND TIME COORDS AUTOMATICALLY
        for col in df.columns:
            # DECODING TIMES IF PRESENT
            if ('time' in col.lower()) | ('_secs' in col.lower()):
                from xarray.conventions import decode_cf_datetime
                time = col
                self.__data__.time_name = time
                nco = Dataset(reference_file_name)
                units = nco.variables[time].getncattr('units')
                df[time] = decode_cf_datetime(df.loc[:, time], units)
                nco.close()

            # INDEXING DIVES IF DEPTH PRESENT
            if 'depth' in col.lower():
                depth = col
                self.__data__.depth_name = col
                # INDEX UP AND DOWN DIVES
                grp = df.groupby('dives')
                dmax = grp[depth].apply(np.nanargmax)
                idx = [grp.groups[i][dmax[i]] for i in grp.groups]
                # create a dummy index 'up' that marks max depth
                df['up'] = 0
                df.loc[idx, 'up'] = 1
                df['up'] = df.up.cumsum()
                # average 'dives' and 'up' for a dive index
                df['dives'] = (df['dives'] + df['up']) / 2.
                df = df.drop('up', axis=1)

        return df

    def bin_depths(self, bins=None, how='mean', depth_name=None):
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
        from scipy.stats import mode

        # load the data if this has not been done
        self.load()

        if depth_name is None:
            if hasattr(self.__data__, 'depth_name'):
                depth = self.__data__.depth_name
            else:
                print('please provide the name of the depth column')
                return None

        if bins is None:
            # this is the CSIR default that is chosen to save battery
            bins = np.arange(0, 1000)
        self.bins = bins

        if bins.max() < self.data[depth].max():
            step = mode(np.diff(bins[-10:])).mode
            start = bins[-1] + step
            stop = self.data[depth].max() + step
            extended_bins = np.arange(start, stop, step)
            bins = np.r_[bins, extended_bins]

        labels = (bins[:-1] + bins[1:]) / 2.
        dives = self.data.dives
        bins = cut(self.data[depth], bins, labels=labels)

        grp = self.data.groupby([dives, bins])
        grp_agg = getattr(grp[self.name], how)()
        gridded = grp_agg.unstack(level=0)

        grp_cols = self.data[self.name].groupby(dives)
        grp_idx0 = [grp_cols.groups[k][0] for k in grp_cols.groups]
        if hasattr(self.__data__, 'time_name'):
            gridded.columns = self.__data__.loc[grp_idx0, self.__data__.time_name]
        else:
            gridded.columns = self.data.loc[grp_idx0, 'dives']

        self.__gridded__ = gridded

        return gridded

    def scatter(self, **kwargs):
        """
        Plot a scatter plot of the dives with x-time and y-depth and
        c-variable.
        The **kwargs can be anything that gets passed to plt.scatter.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from matplotlib.pyplot import scatter, colorbar, subplots

        self.load()
        time = getattr(self.__data__, 'time_name', None)
        if time is not None:
            x = self.data.set_index(time).index.to_pydatetime()
        else:
            x = self.data.dives
        y = self.data[self.__data__.depth_name]
        z = np.ma.masked_invalid(self.data[self.name])

        if "vmin" not in kwargs:
            kwargs['vmin'] = np.percentile(z[~z.mask], 1)
        if "vmax" not in kwargs:
            kwargs['vmax'] = np.percentile(z[~z.mask], 99)
        if 'linewidths' not in kwargs:
            kwargs['linewidths'] = 0

        fig, ax = subplots(1, 1, figsize=[11, 4])

        im = scatter(x, y, 20, z, **kwargs)
        cb = colorbar(mappable=im, pad=0.02, ax=ax)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')

        vname = self.name.capitalize()
        units = '({})'.format(
            self.attrs['units']) if 'units' in self.attrs else ''
        clabel = '{} {}'.format(vname, units)
        cb.set_label(clabel, rotation=-90, va='bottom')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    def pcolormesh(self, interpolated=True, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable.
        The **kwargs can be anything that gets passed to plt.pcolormesh.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from matplotlib.pyplot import pcolormesh, colorbar, subplots

        if interpolated:
            gridded = self.gridded.interpolate(method='linear', limit=4)
        else:
            gridded = self.gridded

        y = gridded.index.values
        x = gridded.columns.values
        z = np.ma.masked_invalid(gridded)

        if "vmin" not in kwargs:
            kwargs['vmin'] = np.percentile(z[~z.mask], 1)
        if "vmax" not in kwargs:
            kwargs['vmax'] = np.percentile(z[~z.mask], 99)

        fig, ax = subplots(1, 1, figsize=[11, 4])
        im = pcolormesh(x, y, z, **kwargs)
        cb = colorbar(mappable=im, pad=0.02, ax=ax)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')

        vname = self.name.capitalize()
        units = '({})'.format(
            self.attrs['units']) if 'units' in self.attrs else ''
        clabel = '{} {}'.format(vname, units)
        cb.set_label(clabel, rotation=-90, va='bottom')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax


class _SeaGliderPointVariable:
    def __init__(self, name, files):

        nco = Dataset(files[0])
        self.__files__ = files
        self.__data__ = None
        self.name = name
        self.attrs = dict(nco.variables[name].__dict__)
        nco.close()

    @property
    def data(self):
        if self.__data__ is None:
            self.load()
        return self.__data__

    def load(self):
        if self.__data__ is None:
            df = self._read_nc_files(self.__files__, self.name)
            try:
                self.__data__ = df.astype(float)
            except ValueError:
                self.__data__ = df

    def __repr__(self):
        string = ""
        string += "Variable:         {}\n".format(self.name)
        string += "Number of dives:  {}\n".format(self.__files__.size)

        string += "Attributes:\n"
        if self.attrs:
            for key in self.attrs:
                string += '    {0: <18}{1}\n'.format(key.capitalize() + ":", self.attrs[key])
        else:
            string += "     No attributes for this variable\n"

        string += '\n'
        if self.__data__ is not None:
            string += "Data is not interpolated/binned \n"
            string += self.data.head().__repr__()
            string += "\n{} measurements".format(self.data.shape[0])
        else:
            string += 'Data not loaded yet [.data]'

        return string

    def _read_nc_files(self, files, key):
        from tqdm import trange

        data = []
        for i in trange(files.size):
            fname = files[i]
            nc = Dataset(fname)
            arr = nc.variables[key][:].squeeze()

            if arr.size == 1:
                pass
            else:
                arr = ''.join(arr.astype(str))
            nc.close()
            data += arr,

        df = Series(np.array(data), name=key)

        return df


class plotting:
    def depth_binning(depth, bins=None):
        from numpy import diff, abs, linspace
        from matplotlib.pyplot import subplots, colorbar, cm
        from matplotlib.colors import LogNorm

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

        if bins is not None:
            xb = abs(diff(bins))
            yb = bins[1:]
            ax.plot(xb, yb, color='orange', lw=4)

        return ax


class processing:
    from seawater import dens as calc_density

    def flr_quenching(chl, ):
        pass

    def flr_dark_count(df):
        pass

    def par_scaling(par, ):
        pass

    def par_dark_count(df):
        pass

    def bks_scaling(bb):
        pass

    def bks_dark_count(bb):
        pass

    def bks_zang2009(bb):
        pass

    def bks_chi(bb):
        pass

    def bottle_calibration(df):
        pass

    def despike_running_median(df):
        pass

    def despike_median_curve(df):
        pass

    def despike_running_mean(df):
        pass


if __name__ == '__main__':
    pass
