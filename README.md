# OSM Plotter

A Python project for processing and visualizing OpenStreetMap data, specifically focused on the German transport network.

## Setup

1. Ensure you have Python 3.8 or higher installed
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Project Structure

- `src/`: Source code files
  - `germany_transport_network.py`: Main script for processing transport network data
  - `operator_mapping.py`: Mapping definitions for operators
- `data/`: Input data files
  - OpenStreetMap data (*.pbf)
  - Classification maps (*.tif)
  - Census data (*.tif)
  - Terrain data (*.tif)
- `cache/`: Cached processed data
- `config/`: Configuration files
  - `config.yaml`: Main configuration settings
- `output/`: Generated output files and logs

## Data Files

The project uses several data files that will be automatically downloaded when you first run the script. The files include:

1. **OpenStreetMap Data (Germany)**
   - File: `germany-latest.osm.pbf`
   - Source: [Geofabrik Germany Extract](https://download.geofabrik.de/europe/germany-latest.osm.pbf)

2. **Population Density Data**
   - File: `Zensus_Bevoelkerung_100m-Gitter.tif`
   - Source: [Census Population Grid](http://hannes.enjoys.it/opendata/Zensus_Bevoelkerung_100m-Gitter.tif)

3. **Land Use Classification Data**
   - File: `classification_map_germany_2020_v02.tif`
   - Source: [MUNDIALIS Land Use Classification](https://data.mundialis.de/geodata/lulc-germany/classification_2020/classification_map_germany_2020_v02.tif)

4. **Terrain Data (SRTM)**
   - File: `srtm_germany_dtm.zip` (will be extracted automatically)
   - Source: [FU Berlin SRTM Data](https://userpage.fu-berlin.de/soga/data/raw-data/spatial/srtm_germany_dtm.zip)

All files will be downloaded automatically to the `data/` directory when needed. You can also manually download them using:

```bash
python src/download_data.py
```

### Cache Files

The following cache files will be automatically generated in the `cache/` directory:
- `processed_roads.json`
- `processed_autobahnen.json`
- `processed_landstrassen.json`
- `processed_railways.json`
- `processed_power_lines.json`

To regenerate cache files, use the `clear_cache()` function.
