from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.tools as tls
import pandas as pd
import xarray as xr
from plotly.express import choropleth
import cartopy.crs as ccrs
from branca.element import Template, MacroElement

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.offline as pio
import plotly.graph_objects as go
from cartopy import crs as ccrs, feature as cfeature
from glob import glob
import folium
from folium import plugins
import numpy as np
import cmasher as cmr
import dash_leaflet as dl


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


def plot_html(date):
    vmin = 0
    vmax = 2
    region = dict(latitude=slice(40, 60), longitude=slice(-50, -20))  # UK

    xs_file = [x for x in xs_files if date in x][0]
    ys_file = [x for x in ys_files if date in x][0]
    ftle_file = [x for x in ftle_files if date in x][0]
    xs = xr.open_dataarray(xs_file)
    ys = xr.open_dataarray(ys_file)
    ftle = xr.open_dataarray(ftle_file)
    cs = ftle

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
        data.append(np.column_stack([y.values.tolist(), x.values.tolist(), cs__.values.tolist()]).tolist())

    color = create_hexcolor_gradient('cmr.gem', n_intervals=30)
    m = folium.Map([35, -30], zoom_start=2.7, tiles='cartodbdark_matter')
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
    m.save(f'ftle_bronagh_{date}.html')
    return m


def produce_maps(date):
    import os
    if not os.path.exists(f'ftle_bronagh_{date}.html'):
        plot_html(date)
    html = open(f'ftle_bronagh_{date}.html', 'r').read()
    return html


if __name__ == '__main__':


    inpath = '/home/gab/phd/data/'

    xs_files = glob(inpath + '*xs_2018*.nc')
    ys_files = glob(inpath + '*ys_2018*.nc')
    ftle_files = glob(inpath + '*ftle_2018*.nc')
    available_dates = [x.split('/')[-1].split('_')[1].split('.')[0] for x in ftle_files]
    available_dates.sort()
    # produce_maps()



    app = Dash(__name__)
    Div_Header = html.Div(children=[html.H1(children='FTLE and trajectories for Storm Bronagh - September 2018',
                                            style={'font-family': 'ubuntu', 'text-align': 'center'})])
    Div_Mapa = html.Div([
        html.Iframe(id='map',
                    style={'width': '100%', 'padding-left':'80px', 'height': '99%', 'position':'absolute'})
        ]
    )
    Div_Slider = html.Div([
        dcc.Slider(min=0, max=len(available_dates)-1, step=1,
                   value=0,
                   marks=dict(zip([str(x) for x in np.arange(len(available_dates)).tolist()],
                                  [pd.Timestamp(x).strftime('%m/%dT%H') for x in available_dates])),
                   id='my-slider', vertical=True
                   ),
        html.Div(id='slider-output-container'),
    ] )
    app.layout = html.Div([Div_Header, Div_Mapa, Div_Slider], style={"background-color": 'white', 'height': '88%', 'position':'absolute',
                                              'width': '88%'})

    @app.callback(
        Output('map', 'srcDoc'),
        Input('my-slider', 'value'))
    def update_output(value):
        date = available_dates[value]
        html = produce_maps(date)

        return html


    #
    #
    #
    #     return plotly_fig
    app.run_server(debug=True)