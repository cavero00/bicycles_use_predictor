import folium
import geopandas as gpd

# Crear mapa base centrado en Chicago
m = folium.Map(location=[41.8781, -87.6298], zoom_start=12)

# Cargar capa 1 (polígonos verdes)
gdf1 = gpd.read_file("bikes.geojson")
folium.GeoJson(
    gdf1,
    style_function=lambda x: {
        'fillColor': 'green',
        'color': 'black',
        'weight': 2,
        'fillOpacity': 0.5
    }
).add_to(m)

# Cargar capa 2 (puntos naranjas con nombre visible)
gdf2 = gpd.read_file("good_weather_stations.geojson")

for _, row in gdf2.iterrows():
    lon, lat = row.geometry.x, row.geometry.y
    name = row["name"]   # propiedad del GeoJSON
    
    # Usamos popup con sticky=True para que se quede siempre visible
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(name, max_width=600, sticky=True),  # Se mantiene visible
        icon=folium.Icon(color='orange', icon='info-sign')
    ).add_to(m)

# Guardar mapa combinado
m.save("mapa_combinado.html")
