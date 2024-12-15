import os
import json
import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.plot import show
from scipy.ndimage import gaussian_filter
from rasterio.warp import calculate_default_transform, reproject, Resampling
import colorsys
import matplotlib.colors
from operator_mapping import OPERATOR_MAPPING
import osmnx as ox
import osmium
import math
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/api_queries.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure osmnx settings
ox.settings.use_cache = True
ox.settings.log_console = True

# Initialize global cache
_data_cache = {
    'roads': None,
    'highways': None,
    'federal_roads': None,
    'railways': None,
    'power_lines': None
}

def load_cached_data(data_type):
    """Load data from cache if available, otherwise load from file"""
    if _data_cache[data_type] is None:
        logger.info(f"Loading {data_type} from cache...")
        _data_cache[data_type] = load_from_cache(data_type)
        logger.info(f"Loaded {len(_data_cache[data_type])} {data_type} features")
    return _data_cache[data_type]

def load_from_cache(data_type):
    cache_files = {
        'roads': 'cache/processed_roads.json',
        'highways': 'cache/processed_autobahnen.json',
        'federal_roads': 'cache/processed_landstrassen.json',
        'railways': 'cache/processed_railways.json',
        'power_lines': 'cache/processed_power_lines.json'
    }
    
    cache_file = cache_files[data_type]
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        features = []
        for item in tqdm(cached_data['features'], desc=f"Loading cached {data_type}"):
            coords = item['coordinates']
            if len(coords) >= 2:
                valid_coords = [(x, y) for x, y in coords if not math.isinf(x) and not math.isinf(y)]
                if len(valid_coords) >= 2:
                    feature_data = {
                        'geometry': LineString(valid_coords),
                        'id': item['id']
                    }
                    for key in ['highway', 'lanes', 'ref', 'name', 'railway', 'voltage']:
                        if key in item:
                            feature_data[key] = item[key]
                    features.append(feature_data)
        if features:
            gdf = gpd.GeoDataFrame(features)
            gdf.set_geometry('geometry', inplace=True)
            gdf.set_crs('EPSG:4326', inplace=True)
            gdf = gdf[gdf.geometry.is_valid]
            if len(gdf) > 0:
                gdf = gdf.to_crs('EPSG:25832')
                return gdf
            else:
                return None
        else:
            return None
    else:
        return None

class UnifiedTransportHandler(osmium.SimpleHandler):
    def __init__(self):
        super(UnifiedTransportHandler, self).__init__()
        self.ways = {
            'roads': [],
            'highways': [],
            'federal_roads': [],
            'railways': [],
            'power_lines': []  # Add power lines
        }
        self.debug_counter = 0
        
    def way(self, w):
        """Process all transportation types in a single pass"""
        self.debug_counter += 1
        if self.debug_counter % 100000 == 0:
            counts = {k: len(v) for k, v in self.ways.items()}
            logger.info(f"Processed {self.debug_counter} ways, found: {counts}")
        
        # Early filtering - only process ways we're interested in
        if not any(tag in w.tags for tag in ['highway', 'railway', 'power']):
            return
            
        # For highways, only process specific types
        if 'highway' in w.tags:
            highway = w.tags.get('highway')
            if highway not in ['motorway', 'trunk', 'residential', 'tertiary']:
                return
                    
        # For railways, only process specific types
        if 'railway' in w.tags:
            railway = w.tags.get('railway')
            if railway not in ['rail', 'light_rail', 'subway', 'tram']:
                return
                
        # For power lines, only process high voltage lines
        if 'power' in w.tags:
            if w.tags.get('power') != 'line':
                return
            voltage = w.tags.get('voltage', '')
            if not any(v in voltage for v in ['380000', '220000']):
                return
        
        # Get coordinates - do this last since it's expensive
        coords = []
        for n in w.nodes:
            lon, lat = n.location.lon, n.location.lat
            # Ensure coordinates are within reasonable bounds for Germany
            if 5.0 <= lon <= 15.0 and 47.0 <= lat <= 55.0:
                coords.append((lon, lat))
        
        if len(coords) < 2:
            return
            
        try:
            line = LineString(coords)
            if not line.is_valid:
                return
                
            feature_data = {
                'geometry': line,
                'id': w.id,
                'coordinates': coords,  # Store original coordinates for caching
                'name': w.tags.get('name', ''),
                'ref': w.tags.get('ref', '')
            }
            
            # Categorize the way
            if 'railway' in w.tags:
                feature_data['railway'] = w.tags.get('railway')
                self.ways['railways'].append(feature_data)
            elif 'power' in w.tags:
                feature_data['voltage'] = w.tags.get('voltage', '')
                self.ways['power_lines'].append(feature_data)
            else:  # Must be a highway since we filtered earlier
                highway = w.tags.get('highway')
                feature_data['highway'] = highway
                feature_data['lanes'] = w.tags.get('lanes', '1')
                
                if highway == 'motorway':
                    self.ways['highways'].append(feature_data)
                elif highway == 'trunk':
                    self.ways['federal_roads'].append(feature_data)
                else:  # residential or tertiary roads
                    self.ways['roads'].append(feature_data)
                    
        except Exception as e:
            logger.warning(f"Failed to process way {w.id}: {e}")

def process_transport_infrastructure(osm_file):
    """Process all transportation infrastructure in a single pass"""
    logger.info("Processing transportation infrastructure...")
    
    # Check if all cache files exist
    cache_files = {
        'roads': 'cache/processed_roads.json',
        'highways': 'cache/processed_autobahnen.json',
        'federal_roads': 'cache/processed_landstrassen.json',
        'railways': 'cache/processed_railways.json',
        'power_lines': 'cache/processed_power_lines.json'
    }
    
    all_cached = all(os.path.exists(f) for f in cache_files.values())
    if all_cached:
        results = {}
        for infra_type, cache_file in cache_files.items():
            logger.info(f"Loading {infra_type} from cache...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            features = []
            for item in tqdm(cached_data['features'], desc=f"Loading cached {infra_type}"):
                coords = item['coordinates']
                if len(coords) >= 2:
                    valid_coords = [(x, y) for x, y in coords if not math.isinf(x) and not math.isinf(y)]
                    if len(valid_coords) >= 2:
                        feature_data = {
                            'geometry': LineString(valid_coords),
                            'id': item['id']
                        }
                        for key in ['highway', 'lanes', 'ref', 'name', 'railway', 'voltage']:
                            if key in item:
                                feature_data[key] = item[key]
                        features.append(feature_data)
            
            if features:
                gdf = gpd.GeoDataFrame(features)
                gdf.set_geometry('geometry', inplace=True)
                gdf.set_crs('EPSG:4326', inplace=True)
                gdf = gdf[gdf.geometry.is_valid]
                if len(gdf) > 0:
                    gdf = gdf.to_crs('EPSG:25832')
                    results[infra_type] = gdf
                    logger.info(f"Loaded {len(gdf)} {infra_type} features")
                else:
                    results[infra_type] = None
            else:
                results[infra_type] = None
        
        return results
    
    # If cache doesn't exist, process the OSM file
    handler = UnifiedTransportHandler()
    logger.info("Processing OSM file (this might take a few minutes)...")
    handler.apply_file(osm_file, locations=True)
    
    results = {}
    for infra_type, ways in handler.ways.items():
        if ways:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(ways)
            gdf.set_geometry('geometry', inplace=True)
            gdf.set_crs('EPSG:4326', inplace=True)
            gdf = gdf.to_crs('EPSG:25832')
            results[infra_type] = gdf
            
            # Cache the processed data
            cache_file = cache_files[infra_type]
            features = []
            for _, row in gdf.iterrows():
                feature = {
                    'coordinates': row['coordinates'],
                    'id': row['id']
                }
                for key in ['highway', 'lanes', 'ref', 'name', 'railway', 'voltage']:
                    if key in row:
                        feature[key] = row[key]
                features.append(feature)
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({'features': features}, f)
            
            logger.info(f"Cached {len(features)} {infra_type} features")
        else:
            results[infra_type] = None
            
    return results

def get_public_transport():
    """Get public transport lines from local OSM file"""
    logger.info("Processing public transport...")
    infrastructure = process_transport_infrastructure('germany-latest.osm.pbf')
    railways = infrastructure.get('railways', [])
    
    if railways is None or len(railways) == 0:
        return gpd.GeoDataFrame()
        
    # Filter for urban rail systems (subway, light rail, tram)
    urban_rail = railways[railways['railway'].isin(['subway', 'light_rail', 'tram'])]
    
    if len(urban_rail) == 0:
        return gpd.GeoDataFrame()
        
    return urban_rail

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            'figure': {
                'width': 20,
                'dpi': 300,
                'bbox_inches': 'tight',
                'pad_inches': 0.1
            },
            'layers': {
                'population_density': {
                    'enabled': False,
                    'smoothing_sigma': 5,
                    'high_density_color': '#FFFFFF',
                    'low_density_alpha': 0.1,
                    'style': 'sharp'
                },
                'states': {
                    'enabled': True,
                    'color': '#333333',
                    'line_width': 0.5,
                    'alpha': 0.5
                },
                'federal_roads': {
                    'enabled': True,
                    'color': '#8B4513',
                    'line_width': 0.5,
                    'alpha': 0.6
                },
                'highways': {
                    'enabled': True,
                    'color': '#4169E1',
                    'line_width': 0.75,
                    'alpha': 0.7
                },
                'national_railroads': {
                    'enabled': True,
                    'color': '#32CD32',
                    'line_width': 0.5,
                    'alpha': 0.6
                },
                'public_transport': {
                    'enabled': True,
                    'color': '#00FFFF',
                    'line_width': 0.3,
                    'alpha': 0.5
                },
                'power_lines': {
                    'enabled': True,
                    'color': '#FFA07A',
                    'line_width': 0.2,
                    'alpha': 0.6,
                    'colors': {
                        '380kv': '#FF0000',
                        '220kv': '#00FF00'
                    }
                },
                'elevation': {
                    'enabled': True,
                    'file': 'data/srtm_33_03.tif',
                    'color_ramp': 'terrain',
                    'alpha': 0.3,
                    'smoothing_sigma': 0
                },
                'classification': {
                    'enabled': True,
                    'file': 'data/classification.tif',
                    'colors': {
                        '0': '#000000',
                        '1': '#00FF00',
                        '2': '#0000FF',
                        '3': '#FF0000'
                    },
                    'alpha': 0.5
                }
            }
        }

def create_transport_overview(config_path='config/config.yaml', variant='default'):
    """Create a comprehensive transport network overview based on config"""
    config = load_config(config_path)
    
    # Select the configuration variant
    if variant == 'inverted':
        config = config['inverted']
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Set up the figure
    plt.style.use('dark_background' if variant != 'inverted' else 'default')
    fig_cfg = config['figure']
    fig, ax = plt.subplots(figsize=(fig_cfg['width'], fig_cfg['width']), dpi=fig_cfg.get('dpi', 300))
    
    # Load state boundaries first to calculate proper aspect ratio
    logger.info("Loading state boundaries...")
    states = gpd.read_file("https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json")
    states = states.to_crs(epsg=25832)
    
    # Ensure proper aspect ratio and coverage
    bounds = states.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    aspect_ratio = height / width

    # Adjust figure size to match Germany's proportions
    fig_width = config['figure']['width']
    fig_height = fig_width * aspect_ratio

    # Create figure with appropriate background
    fig = plt.figure(figsize=(fig_width, fig_height))
    if variant != 'inverted':
        fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111)
    if variant != 'inverted':
        ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')

    # Set the plot extent based on states boundaries
    x_min, y_min, x_max, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Process elevation data first if enabled
    if config['layers']['elevation'].get('enabled', False):
        elevation_data = process_elevation_data(config['layers']['elevation'])
        if elevation_data is not None:
            # Create colormap for elevation
            cmap = plt.get_cmap(config['layers']['elevation'].get('color_ramp', 'terrain'))
            # Normalize and plot elevation
            norm = plt.Normalize(vmin=elevation_data.min(), vmax=elevation_data.max())
            ax.imshow(elevation_data, cmap=cmap, 
                     alpha=config['layers']['elevation'].get('alpha', 0.3),
                     extent=[x_min, x_max, y_min, y_max],
                     transform=ax.transData)
            logger.info("Successfully plotted elevation data")
    
    # Process classification data if enabled
    if config['layers'].get('classification', {}).get('enabled', False):
        classification_data = process_classification_data(config['layers']['classification'])
        if classification_data is not None:
            logger.info("Plotting land use classification...")
            class_cfg = config['layers']['classification']
            
            # Create colormap from class colors
            max_class = int(classification_data.max())  # Convert to int to avoid overflow
            colors = [class_cfg['colors'].get(str(i), '#000000') for i in range(max_class + 1)]
            cmap = matplotlib.colors.ListedColormap(colors)
            
            # Plot classification with explicit extent matching the state boundaries
            ax.imshow(classification_data, 
                     cmap=cmap,
                     alpha=class_cfg.get('alpha', 0.5),
                     extent=[x_min, x_max, y_min, y_max],
                     interpolation='nearest',
                     zorder=1)  # Set zorder to ensure it's plotted below transport layers
            logger.info(f"Classification plot extent: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
            logger.info("Successfully plotted land use classification")
    
    # Process and plot population density if enabled and file exists
    if config['layers']['population_density']['enabled']:
        try:
            pop_density_result = process_population_density(config['layers']['population_density']['smoothing_sigma'])
            if pop_density_result is not None:
                plot_population_density(ax, pop_density_result, config['layers']['population_density'])
            else:
                logger.warning("Population density data processing returned None")
        except Exception as e:
            logger.warning(f"Failed to process population density data: {str(e)}")
    
    # Plot state boundaries if enabled
    if config['layers']['states']['enabled']:
        logger.info("Plotting state boundaries...")
        states_cfg = config['layers']['states']
        states.boundary.plot(
            ax=ax,
            color=states_cfg['color'],
            linewidth=states_cfg['line_width'],
            alpha=states_cfg['alpha']
        )
        logger.info("Successfully plotted state boundaries")

    # Fetch all data first
    logger.info("Fetching transportation data...")
    # Get all infrastructure data at once
    all_infrastructure = process_transport_infrastructure('germany-latest.osm.pbf')
    
    roads_data = load_cached_data('roads')
    federal_roads_data = load_cached_data('federal_roads')
    highways_data = load_cached_data('highways')
    railroad_data = load_cached_data('railways')
    public_transport_data = get_public_transport()
    power_lines_data = load_cached_data('power_lines')

    # Plot layers in order from background to foreground
    layers_data = {
        'roads': roads_data,
        'federal_roads': federal_roads_data,
        'highways': highways_data,
        'national_railroads': railroad_data,
        'public_transport': public_transport_data,
        'power_lines': power_lines_data
    }

    # Check bounds after transformation
    for layer_name, data in layers_data.items():
        if data is not None:
            bounds = data.total_bounds
            logger.info(f"{layer_name} bounds after transformation: x=[{bounds[0]}, {bounds[2]}], y=[{bounds[1]}, {bounds[3]}]")

    logger.info(f"Plot bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    # Plot each layer
    for layer_name, data in layers_data.items():
        cfg = config['layers'].get(layer_name)
        if cfg and cfg.get('enabled', True) and data is not None:
            logger.info(f"Plotting {layer_name} - Data: Present")
            bounds = data.total_bounds
            logger.info(f"{layer_name} bounds: x=[{bounds[0]}, {bounds[2]}], y=[{bounds[1]}, {bounds[3]}]")
            
            # Check if any geometries are within the plot bounds
            data_in_bounds = data.cx[x_min:x_max, y_min:y_max]
            logger.info(f"{layer_name} features within plot bounds: {len(data_in_bounds)} out of {len(data)}")
            
            if len(data_in_bounds) > 0:
                # For federal roads, use a higher z-order to ensure visibility
                zorder = 3 if layer_name == 'federal_roads' else None
                if layer_name == 'power_lines':
                    # Split power lines by voltage
                    for _, line in data_in_bounds.iterrows():
                        voltage = line['voltage']
                        if '380000' in voltage:
                            color = cfg['colors']['380kv']
                        else:  # Must be 220kV since we filtered earlier
                            color = cfg['colors']['220kv']
                        
                        coords = line['geometry'].coords
                        x, y = zip(*coords)
                        
                        ax.plot(x, y, color=color, linewidth=cfg['line_width'], 
                               alpha=cfg['alpha'], solid_capstyle='round')
                else:
                    data_in_bounds.plot(
                        ax=ax,
                        color=cfg['color'],
                        linewidth=cfg['line_width'],
                        alpha=cfg['alpha'],
                        zorder=zorder
                    )
                logger.info(f"Successfully plotted {layer_name}")
            else:
                logger.warning(f"No {layer_name} features within plot bounds!")
    
    # Save the plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"transport_network_{variant}_{timestamp}"
    
    # Save as PNG for quick preview
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, 
                dpi=fig_cfg.get('dpi', 300),
                bbox_inches=fig_cfg.get('bbox_inches', 'tight'),
                pad_inches=fig_cfg.get('pad_inches', 0.1),
                facecolor=fig_cfg.get('background', 'black') if variant != 'inverted' else 'white')
    logger.info(f"Saved PNG preview to {png_path}")
    
    # Save as PDF for high quality
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path, 
                format='pdf',
                bbox_inches=fig_cfg.get('bbox_inches', 'tight'),
                pad_inches=fig_cfg.get('pad_inches', 0.1),
                facecolor=fig_cfg.get('background', 'black') if variant != 'inverted' else 'white')
    logger.info(f"Saved PDF to {pdf_path}")
    
    plt.close()
    logger.info("Transport network overview created successfully!")

def process_population_density(sigma_km=5):
    """Process population density data with spatial averaging"""
    import rasterio
    from scipy.ndimage import gaussian_filter
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    # Read the Zensus population density raster
    with rasterio.open("data/Zensus_Bevoelkerung_100m-Gitter.tif") as src:
        src_crs = src.crs
        if src_crs is None:
            src_crs = 'EPSG:3035'
        
        # Read the data
        data = src.read(1)
        
        # Create a mask for valid data (non-negative values)
        valid_mask = data >= 0
        
        # Replace negative values with 0 for processing
        data = np.where(valid_mask, data, 0)
        
        if sigma_km > 0:
            # Apply smoothing only to valid data
            sigma_pixels = sigma_km * 10
            
            # Pad the data to prevent edge effects
            pad_size = int(3 * sigma_pixels)  # 3 sigma padding
            padded_data = np.pad(data, pad_size, mode='reflect')
            padded_mask = np.pad(valid_mask, pad_size, mode='reflect')
            
            # Smooth both the data and the mask
            smoothed_data = gaussian_filter(padded_data, sigma=sigma_pixels)
            smoothed_mask = gaussian_filter(padded_mask.astype(float), sigma=sigma_pixels)
            
            # Normalize by the smoothed mask to correct for edge effects
            smoothed_mask = np.where(smoothed_mask > 0.1, smoothed_mask, 1)  # Avoid division by very small numbers
            normalized_data = smoothed_data / smoothed_mask
            
            # Crop back to original size
            data = normalized_data[pad_size:-pad_size, pad_size:-pad_size]
            
            # Apply the original mask
            data = np.where(valid_mask, data, 0)
        
        # Log transform for better visualization
        data = np.log1p(data)
        
        # Calculate transform for reprojection to EPSG:25832
        dst_crs = 'EPSG:25832'
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs,
            src.width, src.height,
            *src.bounds
        )
        
        # Initialize the output array
        data_25832 = np.zeros((height, width), dtype=data.dtype)
        
        # Reproject to EPSG:25832
        reproject(
            source=data,
            destination=data_25832,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        print(f"Source bounds: {src.bounds}")
        print(f"Output bounds: [{transform.c}, {transform.f + height * transform.e}, {transform.c + width * transform.a}, {transform.f}]")
        
        return data_25832, transform, dst_crs

def process_classification_data(config):
    """Process land use classification data"""
    if not config.get('enabled', False):
        logger.warning("Classification layer is disabled")
        return None
        
    classification_file = config['file']
    if not os.path.exists(classification_file):
        logger.warning(f"Classification file {classification_file} not found")
        return None
        
    logger.info(f"Processing classification data from {classification_file}")
    with rasterio.open(classification_file) as src:
        # Read the data
        data = src.read(1)
        logger.info(f"Classification data shape: {data.shape}, unique values: {np.unique(data)}")
        logger.info(f"Classification data CRS: {src.crs}")
        
        # Reproject to EPSG:25832 if needed
        if src.crs != 'EPSG:25832':
            logger.info("Reprojecting classification data to EPSG:25832")
            dst_crs = 'EPSG:25832'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                src.width, src.height,
                *src.bounds
            )
            
            # Create destination array
            classification = np.zeros((height, width), dtype=data.dtype)
            
            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=classification,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            data = classification
            logger.info(f"Reprojected classification data shape: {data.shape}")
            
    return data

def process_elevation_data(config, target_bounds=None):
    """Process elevation data from SRTM"""
    if not config.get('enabled', False):
        return None
        
    elevation_file = config['file']
    if not os.path.exists(elevation_file):
        logger.warning(f"Elevation file {elevation_file} not found")
        return None
        
    logger.info("Processing elevation data...")
    with rasterio.open(elevation_file) as src:
        # Reproject if needed and clip to target bounds
        if target_bounds is not None:
            # Calculate transform for target CRS (EPSG:25832)
            dst_crs = 'EPSG:25832'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, 
                *src.bounds
            )
            
            # Create destination array
            elevation = np.zeros((height, width), dtype=np.float32)
            
            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=elevation,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
        else:
            elevation = src.read(1)
            
    # Apply smoothing if configured
    sigma = config.get('smoothing_sigma', 0)
    if sigma > 0:
        elevation = gaussian_filter(elevation, sigma=sigma)
        
    return elevation

def create_custom_colormap(high_density_color, low_density_alpha, style_config):
    """Create a custom colormap based on the style configuration."""
    if style_config is None or not isinstance(style_config, dict):
        # Original sharp style
        colors = [(0, 0, 0, 0),  # Start with transparent black
                 (0, 0, 0, low_density_alpha),  # Slightly visible black
                 (*matplotlib.colors.to_rgb(high_density_color), 1.0)]  # End with configured color
        n_bins = 100
    else:
        # Smooth style
        n_bins = style_config.get('gradient_steps', 256)
        midpoint_alpha = style_config.get('midpoint_alpha', 0.2)
        edge_softness = style_config.get('edge_softness', 0.3)
        
        # Create more control points for smoother transition
        colors = [
            (0, 0, 0, 0),  # Start transparent
            (0, 0, 0, low_density_alpha * edge_softness),  # Very soft start
            (0, 0, 0, midpoint_alpha),  # Middle point
            (*matplotlib.colors.to_rgb(high_density_color), midpoint_alpha * 2),  # Start color transition
            (*matplotlib.colors.to_rgb(high_density_color), 1.0)  # Full color
        ]
    
    return matplotlib.colors.LinearSegmentedColormap.from_list('custom_density', colors, N=n_bins)

def plot_population_density(ax, pop_density_result, config):
    """Plot population density data"""
    data, transform, crs = pop_density_result
    
    cmap = create_custom_colormap(
        config['high_density_color'],
        config['low_density_alpha'],
        config.get('smooth_style', {})  # Fixed to use smooth_style instead of style
    )
    
    # Plot the raster data
    extent = [
        transform.c,  # left
        transform.c + data.shape[1] * transform.a,  # right
        transform.f + data.shape[0] * transform.e,  # bottom
        transform.f,  # top
    ]
    ax.imshow(data, extent=extent, cmap=cmap, alpha=config['low_density_alpha'], 
             interpolation='nearest', origin='upper')
    logger.info("Successfully plotted population density")

def clear_cache(preserve_transport=True):
    """Clear cached data
    
    Args:
        preserve_transport (bool): If True, preserve transportation infrastructure caches
    """
    cache_dir = 'cache'
    if os.path.exists(cache_dir):
        transport_caches = {
            'processed_roads.json',
            'processed_autobahnen.json',
            'processed_landstrassen.json',
            'processed_railways.json',
            'processed_power_lines.json'
        }
        
        for file in os.listdir(cache_dir):
            if preserve_transport and file in transport_caches:
                continue
            if file.startswith('processed_'):
                os.remove(os.path.join(cache_dir, file))
                logger.info(f"Removed cache file: {file}")

if __name__ == "__main__":
    # Ensure all required data files are present
    from download_data import ensure_data_files
    ensure_data_files()
    
    # Clear non-transport caches but preserve transport data
    clear_cache(preserve_transport=True)
    create_transport_overview()
