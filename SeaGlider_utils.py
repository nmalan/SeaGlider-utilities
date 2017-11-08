#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UAGE:   from SeaGlider_utils import SeaGlider
        sg = SeaGlider('/path/to/folder/')

For more info see help(SeaGlider)

written by Luke Gregor (lukegre@gmail.com)
"""

from __future__ import print_function as _pf
from netCDF4 import Dataset
from pandas import DataFrame, Series
import numpy as np
import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        if 'coordinates' in self.attrs:
            coordinates = self.attrs['coordinates'].split() + ['dives']
            self.__data__.coordinates = coordinates
        elif not hasattr(self.__data__, 'coordinates'):
            self.__data__.coordinates = []
        self.__data__.bins = np.arange(1000.)

        nco.close()

    def __getitem__(self, key):
        data = self.load(return_data=True)
        return data.loc[key]

    @property
    def gridded(self):
        if self.__gridded__ is None:
            self.__gridded__ = self.bin_depths()
        return self.__gridded__

    @property
    def data(self):

        return self.load(return_data=True)

    @property
    def series(self):
        self.load()
        return self.__data__.loc[:, self.name]

    @property
    def values(self):
        self.load()
        return self.__data__.loc[:, self.name].values

    def load(self, return_data=False):
        # neaten up the script by creating labels
        data = self.__data__
        keys = np.unique(data.coordinates + [self.name])
        files = self.__files__

        # get keys not in dataframe
        missing = [k for k in filter(lambda k: k not in data, keys)]

        if any(missing):
            df = self._read_nc_files(files, missing)
            # process coordinates - if no coordinates, just loops through
            df = self._process_coords(df, files[0])
            for col in df:
                # inplace column creation
                data[col] = df[col]

        if return_data:
            return data.loc[:, keys]

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

    def _read_nc_files(self, files, keys):
        from tqdm import trange

        if 'dives' in keys:
            dives = True
            keys.remove('dives')
        else:
            dives = False

        data = []
        for i in trange(files.size):
            fname = files[i]
            nc = Dataset(fname)
            if any([k not in nc.variables for k in keys]):
                message = "{} is missing some variables".format(fname)
                print(message)
                continue
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
                time = col
                self.__data__.time_name = time
                nco = Dataset(reference_file_name)
                units = nco.variables[time].getncattr('units')
                df[time + '_raw'] = df.loc[:, time].copy()
                if 'seconds since 1970' in units:
                    df[time] = df.loc[:, time].astype('datetime64[s]')
                else:
                    from xarray.conventions import decode_cf_datetime
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

    def bin_depths(self, bins=None, how='mean', depth=None):
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
        data = self.__data__

        if depth is None:
            if hasattr(data, 'depth_name'):
                depth = data.depth_name
            else:
                print('please provide depth column name as a kwarg')
                return None

        # assigning a bin value and make that the default for dimension
        bins = data.bins if bins is None else bins
        data.bins = bins

        # The default bins only go to 1000m.
        # Deepen with step same as last 10 values if needed
        if bins.max() < self.data[depth].max():
            step = mode(np.diff(bins[-10:])).mode
            start = bins[-1] + step
            stop = self.data[depth].max() + step
            extended_bins = np.arange(start, stop, step)
            bins = np.r_[bins, extended_bins]

        #
        labels = (bins[:-1] + bins[1:]) / 2.
        dives = self.data.dives.values
        bins = cut(self.data[depth], bins, labels=labels)

        grp_a = self.data.groupby([dives, bins])
        grp_agg = getattr(grp_a[self.name], how)()
        gridded = grp_agg.unstack(level=0)

        if hasattr(data, 'time_name'):
            grp_b = self.data[depth].groupby(dives)
            idx = grp_b.apply(lambda s: s.index[np.nanargmin(s)])
            gridded.columns = data.loc[idx, data.time_name]
            gridded.columns.name = 'surface_time'

        if gridded.index.dtype.name == 'category':
            index = gridded.index.astype(float)
            gridded = gridded.set_index(index)
            gridded.index.name = 'depth_bin_centre'

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

    def pcolormesh(self, interpolate_dist=0, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable. The data can be linearly interpolated to fill missing
        depth values. The number of points to interpolate can be set with
        interpolate_dist.
        The **kwargs can be anything that gets passed to plt.pcolormesh.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from matplotlib.pyplot import pcolormesh, colorbar, subplots

        if interpolate_dist:
            gridded = self.gridded.interpolate(limit=interpolate_dist)
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


if __name__ == '__main__':
    pass
