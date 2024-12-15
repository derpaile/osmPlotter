import os
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

DATA_SOURCES = {
    'osm': {
        'url': 'https://download.geofabrik.de/europe/germany-latest.osm.pbf',
        'filename': 'germany-latest.osm.pbf'
    },
    'population': {
        'url': 'http://hannes.enjoys.it/opendata/Zensus_Bevoelkerung_100m-Gitter.tif',
        'filename': 'Zensus_Bevoelkerung_100m-Gitter.tif'
    },
    'landuse': {
        'url': 'https://data.mundialis.de/geodata/lulc-germany/classification_2020/classification_map_germany_2020_v02.tif',
        'filename': 'classification_map_germany_2020_v02.tif'
    },
    'elevation': {
        'url': 'https://userpage.fu-berlin.de/soga/data/raw-data/spatial/srtm_germany_dtm.zip',
        'filename': 'srtm_germany_dtm.zip',
        'is_zip': True
    }
}

def download_file(url: str, filename: str, data_dir: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    filepath = data_dir / filename
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def ensure_data_files():
    """Ensure all required data files are present, downloading them if necessary."""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    for source_name, source_info in DATA_SOURCES.items():
        filename = source_info['filename']
        filepath = data_dir / filename
        
        # Check if file needs to be downloaded
        if not filepath.exists():
            print(f"\nDownloading {filename}...")
            try:
                download_file(source_info['url'], filename, data_dir)
                print(f"Successfully downloaded {filename}")
                
                # Extract if it's a zip file
                if source_info.get('is_zip'):
                    print(f"Extracting {filename}...")
                    extract_zip(filepath, data_dir)
                    # Remove zip file after extraction
                    filepath.unlink()
                    print(f"Successfully extracted {filename}")
                    
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                if filepath.exists():
                    filepath.unlink()  # Clean up partial download
                continue
        else:
            print(f"{filename} already exists, skipping download.")

if __name__ == "__main__":
    ensure_data_files()
