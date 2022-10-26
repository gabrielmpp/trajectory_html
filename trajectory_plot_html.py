import xarray as xr
import folium
import folium.plugins as plugins
import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
from branca.element import Template, MacroElement
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from cartopy.io.img_tiles import Stamen
import pandas as pd
from owslib.wmts import WebMapTileService
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
from LagrangianCoherence.LCS.trajectory import parcel_propagation
import matplotlib
from plotlib.miaplot.miaplot import plot
from windspharm.xarray import VectorWind

def colorbar(vmin, vmax, cmap, ndigits=1, title='', vertical_padding=0):
    """

    @param ndigits: Number of decimals in the cbar
    @param title: String with title for cbar
    @param vmin: float minimum value for cbar
    @param vmax: float maximum value for cbar
    @param cmap: matplotlib colormap
    @return: str with html code
    """
    intervals = np.linspace(vmin, vmax, 11).tolist()
    intervals = [str(round(x, ndigits=ndigits)) for x in intervals]
    cmap = matplotlib.cm.get_cmap(cmap, len(intervals))
    colors = []
    for i, interval in enumerate(intervals):
        rgb = cmap(i)[:3]
        colors.append(str(matplotlib.colors.rgb2hex(rgb)))  # add i+1 if colors[0]=black is uncommented
    colors = [colors, intervals]
    # colors[1][2]
    position_bottom_colorbar = str(20 + vertical_padding)

    return '''

        {% macro html(this, kwargs) %}

        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>jQuery UI Draggable - Default functionality</title>
          <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

          <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
          <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

          <script>
          $( function() {
            $( "#colorbar" ).draggable({
                            start: function (event, ui) {
                                $(this).css({
                                    right: "auto",
                                    top: "auto",
                                    bottom: "auto"
                                });
                            }
                        });
        });

          </script>
        </head>
        <body>



        <div id='colorbar' class='my-legend'
            style='position: fixed; bottom: ''' + position_bottom_colorbar + '''px; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.5);
             border-radius:0px; padding: 4px; font-size:13px; right: 20px; '>
        <div class='legend-title'>''' + title + '''</div>
        <div class='legend-scale'>
          <ul class='legend-labels'>

            <li><span style='background:''' + colors[0][0] + ''';'></span>''' + colors[1][0] + '''</li>
            <li><span style='background:''' + colors[0][1] + ''';'></span>''' + colors[1][1] + '''</li>
            <li><span style='background:''' + colors[0][2] + ''';'></span> ''' + colors[1][2] + '''</li>

            <li><span style='background:''' + colors[0][3] + ''';'></span>''' + colors[1][3] + '''</li>
            <li><span style='background:''' + colors[0][4] + ''';'></span>''' + colors[1][4] + '''</li>
            <li><span style='background:''' + colors[0][5] + ''';'></span>''' + colors[1][5] + '''</li>
            <li><span style='background:''' + colors[0][6] + ''';'></span>''' + colors[1][6] + '''</li>

            <li><span style='background:''' + colors[0][7] + ''';'></span>''' + colors[1][7] + '''</li>
            <li><span style='background:''' + colors[0][8] + ''';'></span>''' + colors[1][8] + '''</li>
            <li><span style='background:''' + colors[0][9] + ''';'></span>''' + colors[1][9] + '''</li>
            <li><span style='background:''' + colors[0][10] + ''';'></span>''' + colors[1][10] + '''</li>

          </ul>
        </div>
        </body>
        </html>
        <style type='text/css'>
          .my-legend .legend-title {
            text-align: left;
            margin-bottom: 8px;
            font-weight: bold;
            font-size: 90%;
            }
          .my-legend .legend-scale ul {
            margin: 0;
            padding: 0;
            float: left;
            list-style: none;
            }
          .my-legend .legend-scale ul li {
            display: block;
            float: left;
            width: 50px;
            margin-bottom: 2px;
            text-align: center;
            font-size: 80%;
            list-style: none;
            }
          .my-legend ul.legend-labels li span {
            display: block;
            float: left;
            height: 15px;
            width: 50px;
            }
          .my-legend .legend-source {
            font-size: 70%;
            color: #999;
            clear: both;
            }
          .my-legend a {
            color: #777;
            }
        </style>
        {% endmacro %}

    '''

def create_hexcolor_gradient(cmap='RdBu', n_intervals=10):
    intervals = np.linspace(0, 1, n_intervals).tolist()

    cmap = matplotlib.cm.get_cmap(cmap, len(intervals))
    colors = dict.fromkeys(intervals)
    for i, interval in enumerate(colors.keys()):
        rgb = cmap(i)[:3]
        colors[interval] = str(matplotlib.colors.rgb2hex(rgb))
    return colors

def calc_ftle(da_urho_, da_vrho_, truncation=20):
    print('********************************** \n'
          'Truncating wind and computing FTLE\n'
          '**********************************')
    w = VectorWind(da_urho_, da_vrho_)
    da_urho_ = w.truncate(da_urho_, truncation=20).resample(time='3H').interpolate()
    da_vrho_ = w.truncate(da_vrho_, truncation=20).resample(time='3H').interpolate()
    print(da_urho_)
    lcs = LCS(timestep=-2 * 3600, timedim='time', SETTLS_order=4)
    ftle, x_trajs, y_trajs = lcs(u=da_urho_, v=da_vrho_, isglobal=True, s=4e6,
               interp_to_common_grid=True, traj_interp_order=3, return_traj=True)
    ftle = .5 * np.log(ftle)
    return ftle, x_trajs, y_trajs


# ---- Preparing input ---- #
outpath = '/home/gab/phd/data/'
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2018010100-2018123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2018010100-2018123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2018010100-2018123118.nc'
pr_filepath = basepath + 'pr_ERA5_6hr_2018010100-2018123118.nc'
data_dir = '/home/gab/phd/data/composites_cz/'
ds_seasonal_avgs = xr.open_dataset(data_dir + 'ds_seasonal_avgs.nc')

# timesel = sys.argv[1]

u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')


# u = u.sel(latitude=slice(-60, 60), longitude=slice(-130, 45))

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
# v = v.sel(latitude=slice(-60, 60), longitude=slice(-130, 45)).sel(expver=1).drop('expver')


tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
# tcwv = tcwv.sel(latitude=slice(-60, 60), longitude=slice(-130, 45)).sel(expver=1).drop('expver')

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'

pts_x_list = []
pts_y_list = []
dt = 12 # SACZ
dt = 90 # BRONAGH

for dt in np.arange(96, 106):
    timeseq = np.arange(0, 8) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    ds = ds.load()
    # ax.add_feature(cfeature.BORDERS)
    # ax.add_feature(cfeature.OCEAN)
    # ax.background_img('BM', resolution='low')
    # ax.add_feature(states_provinces, edgecolor='gray')
    #
    # ax.set_title('Smoke trajectories at ' + pd.Timestamp(ds.time.values[-1]).strftime('%d-%m-%Y'),
    #              x=0.5, y=0.9, color='white')

    ftle, xs, ys = calc_ftle(ds.u, ds.v)
    ftle.to_netcdf(outpath + 'ftle_' + pd.Timestamp(ftle.time.values[0]).strftime('%Y-%m-%dT%H:00') + '.nc')
    xs.to_netcdf(outpath + 'xs_' + pd.Timestamp(ftle.time.values[0]).strftime('%Y-%m-%dT%H:00') + '.nc')
    ys.to_netcdf(outpath + 'ys_' + pd.Timestamp(ftle.time.values[0]).strftime('%Y-%m-%dT%H:00') + '.nc')

xs = xs.unstack()
ys = ys.unstack()
cs = ftle.unstack()
vmin=0
vmax=2
region = dict(latitude=slice(-20, 0), longitude=slice(-70, -40))
region = dict(latitude=slice(40, 60), longitude=slice(-50, -20))  # UK
cs_ = cs.sel(**region)
cs_ = cs_.where(cs_ > vmin, vmin)
cs_ = cs_.where(cs_ < vmax, vmax)

xs_ = xs.sel(**region)
ys_ = ys.sel(**region)

xs_ = xs_.sortby('time')
ys_ = ys_.sortby('time')
cs_ = cs_.sortby('time')

cs_ = cs_.sel(latitude=xs_.latitude, longitude=ys_.longitude)
cs_ = cs_.stack(points=['latitude', 'longitude'])

xs_ = xs_.stack(points=['latitude', 'longitude'])
ys_ = ys_.stack(points=['latitude', 'longitude'])

data = []


for t in xs.time.values:

    x = xs_.sel(time=t)
    y = ys_.sel(time=t)
    cs__ = cs_.isel(time=0).drop('time')
    x = x.dropna('points')
    y = y.dropna('points')
    cs__ = cs__.dropna('points')
    cs__ = 1e-6 + (cs__ - vmin) / (vmax - vmin)
    data.append(np.column_stack([y.values.tolist(), x.values.tolist(), cs__.values.tolist() ]).tolist())


color = create_hexcolor_gradient('cmr.gem', n_intervals=30)
m = folium.Map([20, -46.5],  zoom_start=5, tiles='cartodbdark_matter')
fig, axs = plt.subplots(1, 1, subplot_kw={'projection': ccrs.GOOGLE_MERCATOR})
ftle.isel(time=0).plot(transform=ccrs.PlateCarree(), add_colorbar=False, ax=axs, vmin=0, cmap=cmr.gem)
axs.set_title('')
plt.savefig('BG_MAP.png', bbox_inches='tight', pad_inches=0)
plt.close()
img = folium.raster_layers.ImageOverlay(
    name="Mercator projection SW",
    image='BG_MAP.png',
    bounds=[[-89.75, -180], [89.75, 179.5]],
    opacity=.5,
    interactive=True,
    cross_origin=False,
    zIndex=-10,
)
img.add_to(m)

plugins.HeatMapWithTime(data, scale_radius=True, radius=.35, gradient=color, min_opacity=1, max_opacity=1).add_to(m)
# plugins.HeatMapWithTime(data, min_opacity=1, max_opacity=1, gradient=color, blur=0,
#                         scale_radius=True, radius=.35).add_to(m)
colors = colorbar(vmin, vmax, 'cmr.gem')
macro = MacroElement()
macro._template = Template(colors)
m.get_root().add_child(macro)
folium.LayerControl().add_to(m)
m.save('ftle_bronagh.html')


vmin=0
vmax=20
xs = xs.unstack()
ys = ys.unstack()
tcwv = tcwv.unstack()
xs_ridges = xs.copy().where(ridges == 1)
ys_ridges = ys.copy().where(ridges == 1)

cs_ = -cs.diff('time', label='lower')
cs_ = cs_.where(cs_ > vmin, vmin)
cs_ = cs_.where(cs_ < vmax, vmax)
cs_ = cs_.sel(latitude=xs.latitude, longitude=xs.longitude, method='nearest')
cs_ridges = cs_.copy().where(ridges==1)


cs_ridges = cs_ridges.stack(points=['latitude', 'longitude'])

xs_ridges = xs_ridges.stack(points=['latitude', 'longitude'])
ys_ridges = ys_ridges.stack(points=['latitude', 'longitude'])

data = []
for t in xs.time.values[1:]:
    x = xs_ridges.sel(time=t)
    y = ys_ridges.sel(time=t)
    cs_ridges_ = cs_ridges.sel(time=t)
    x = x.dropna('points')
    y = y.dropna('points')
    cs_ridges_ = cs_ridges_.dropna('points')
    cs_ridges_ = 1e-6 + (cs_ridges_ - vmin) / (vmax - vmin)
    data.append(np.column_stack([y.values.tolist(), x.values.tolist(), cs_ridges_.values.tolist()]).tolist())


color = create_hexcolor_gradient('cmr.freeze')
m = folium.Map([-23.5, -46.5], zoom_start=10)
plugins.HeatMapWithTime(data, min_opacity=.7, max_opacity=.7, gradient=color, scale_radius=True, radius=.18).add_to(m)
colors = colorbar(vmin, vmax, 'cmr.freeze')
macro = MacroElement()
macro._template = Template(colors)
m.get_root().add_child(macro)
m.save('sinks_czs.html')

tcwv.isel(time=-1).plot()
plt.show()


vmin=0
vmax=20
xs = xs.unstack()
ys = ys.unstack()
tcwv = tcwv.unstack()
xs_ridges = xs.copy()#.where(ridges == 1)
ys_ridges = ys.copy()#.where(ridges == 1)

cs_ = cs.diff('time', label='lower')
cs_ = cs_.where(cs_ > vmin, vmin)
cs_ = cs_.where(cs_ < vmax, vmax)
cs_ = cs_.sel(latitude=xs.latitude, longitude=xs.longitude, method='nearest')
cs_ridges = cs_.copy()#.where(ridges==1)


cs_ridges = cs_ridges.stack(points=['latitude', 'longitude'])

xs_ridges = xs_ridges.stack(points=['latitude', 'longitude'])
ys_ridges = ys_ridges.stack(points=['latitude', 'longitude'])

data = []
for t in xs.time.values[1:]:
    x = xs_ridges.sel(time=t)
    y = ys_ridges.sel(time=t)
    cs_ridges_ = cs_ridges.sel(time=t)
    x = x.dropna('points')
    y = y.dropna('points')
    cs_ridges_ = cs_ridges_.dropna('points')
    cs_ridges_ = 1e-6 + (cs_ridges_ - vmin) / (vmax - vmin)
    data.append(np.column_stack([y.values.tolist(), x.values.tolist(), cs_ridges_.values.tolist()]).tolist())


color = create_hexcolor_gradient('cmr.flamingo')
m = folium.Map([-23.5, -46.5], zoom_start=20)
plugins.HeatMapWithTime(data, min_opacity=.7, max_opacity=.7, gradient=color, scale_radius=True, radius=.18).add_to(m)
colors = colorbar(vmin, vmax, 'cmr.flamingo')
macro = MacroElement()
macro._template = Template(colors)
m.get_root().add_child(macro)
m.save('sources.html')

tcwv.isel(time=-1).plot()
plt.show()
cs.sortby('time').diff('time').sum('time').plot()
plt.show()
cs.diff('time').sum('time').plot()

plt.show()

sinks = -cs.sortby('time').isel(time=slice(-3,-1)).diff('time')
sinks = sinks.where(sinks>0)
sinks.sum('time').plot(vmax=10)
plt.show()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
cs.mean('time').plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=True)
ridges.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap='Reds')
ax.coastlines()
plt.savefig('time_integral.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
(cs.sortby('time').isel(time=-1) - cs.sortby('time').isel(time=0)).plot(ax=ax,
                                                                        transform=ccrs.PlateCarree(),
                                                                        add_colorbar=True,
                                                                        cmap='cmr.pride')
ridges.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap='Greys')
ax.coastlines()
plt.savefig('integral_of_diff.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
(cs.sortby('time').isel(time=-2) - cs.sortby('time').isel(time=-1)).plot(ax=ax, vmax=0,
                                                                        transform=ccrs.PlateCarree(),
                                                                        add_colorbar=True,
                                                                        cmap='cmr.pride')
ridges.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap='Greys')
ax.coastlines()
plt.savefig('advection.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()

for i in range(cs.time.values.shape[0]):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    p = cs.sortby('time').isel(time=i).plot(ax=ax, transform=ccrs.PlateCarree(),
                                        add_colorbar=False, vmin=0, vmax=70)
    ridges.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap='Reds')
    ax.coastlines()
    ax.set_title(cs.time.values[i])

    plt.savefig(f'anim/test{i:02}.png', dpi=600,
                transparent=True, pad_inches=.2,  bbox_inches='tight'
                )
    plt.close()

np.sign(ds.u).isel(time=0).plot()
np.angle()
np.angle(ds.u.isel(time=0), ds.v.isel(time=0).T)
plt.show()
ridges.plot()
ds.u.isel(time=0)

ds.u.isel(time=0).plot(robust=True)
ds.v.isel(time=0).plot(robust=True)
plt.show()

u_ = ds.u.sel(latitude=slice(-40, 10), longitude=slice(-80, -30)).coarsen(latitude=3, longitude=3).mean()
v_ = ds.v.sel(latitude=slice(-40, 10), longitude=slice(-80, -30)).coarsen(latitude=3, longitude=3).mean()

u_ = u_.stack(points=['latitude', 'longitude'])
v_ = v_.stack(points=['latitude', 'longitude'])

for pt in u_.points.values:
    u__ = u_.sel(points=pt)

    xx = xs.isel(time=-1).where(ridges==1)
    yy = ys.isel(time=-1).where(ridges==1)
    xx_pt = u__.points.values.reshape(1)[0][1]
    yy_pt = u__.points.values.reshape(1)[0][0]
    dist = np.sqrt((xx-xx_pt)**2 + (yy - yy_pt)**2)
    dist.where(dist==dist.min(), drop=True)
    dist.plot(cmap=cmr.rainforest)
    plt.show()
    yy.plot()
    print(u__)


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
ftle3.plot(ax=ax, add_colorbar=True, transform=ccrs.PlateCarree(), cmap='cmr.freeze', vmin=.2, vmax=2)
ridges.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap='Greys')
ax.coastlines()
plt.savefig('advection2.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
ftle3.plot(ax=axs[0], add_colorbar=True, transform=ccrs.PlateCarree(), cmap='cmr.freeze', vmin=.2, vmax=2)
ftle3_r.plot(ax=axs[1], add_colorbar=False, transform=ccrs.PlateCarree(), cmap='cmr.flamingo', vmin=.2, vmax=2)
axs[0].coastlines()
axs[1].coastlines()
plt.savefig('advection3.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()
