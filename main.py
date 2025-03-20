import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
from ftplib import FTP
from google.cloud import storage

##############################################
# FTP DOWNLOAD: Retrieve CSV from remote FTP server
##############################################

# FTP credentials and file details.
ftp_server = "ftp.nivoda.net"
ftp_user = "leeladiamondscorporate@gmail.com"
ftp_password = "r[Eu;9NB"
remote_file = "Leela Diamond_labgrown.csv"
# Use a relative path for the downloaded file:
local_file = "Labgrown.csv"  

try:
    with FTP(ftp_server) as ftp:
        ftp.login(user=ftp_user, passwd=ftp_password)
        print("FTP login successful.")
        with open(local_file, "wb") as f:
            ftp.retrbinary("RETR " + remote_file, f.write)
        print(f"Downloaded '{remote_file}' to '{local_file}'.")
except Exception as e:
    print("Error downloading file from FTP:", e)
    exit(1)

##############################################
# PART 1: DATA IMPORT, FILTERING & BALANCED SELECTION
##############################################

# --- Helper Functions for Data Transformation ---

def map_shape(row):
    """
    Map the raw shape (from column 'shape') into one of the 10 allowed shapes.
    - For "SQ EMERALD" or "ASSCHER", return "Asscher".
    - For "CUSHION" or "CUSHION BRILLIANT", return "Cushion".
    - For allowed shapes (ROUND, OVAL, etc.), return the title-case string.
    """
    raw_shape = str(row.get('shape', '')).strip().upper()
    try:
        float(row.get('length', 0))
        float(row.get('width', 0))
    except Exception:
        pass
    if raw_shape in ['SQ EMERALD', 'ASSCHER']:
        return 'Asscher'
    if raw_shape in ['CUSHION', 'CUSHION BRILLIANT']:
        return 'Cushion'
    allowed = ['ROUND', 'OVAL', 'PRINCESS', 'EMERALD', 'MARQUISE', 'PEAR', 'RADIANT', 'HEART']
    if raw_shape in allowed:
        return raw_shape.title()
    return None

def compute_ratio(row):
    """Compute the ratio (length/width) from the row."""
    try:
        l = float(row.get('length', 0))
        w = float(row.get('width', 0))
        if w:
            return l / w
    except Exception:
        pass
    return np.nan

def compute_measurement(row):
    """Return a measurement string in the format 'length x width - height'."""
    return f"{row.get('length', '')} x {row.get('width', '')} - {row.get('height', '')}"

def valid_cut(row):
    """
    For Round diamonds only, validate the 'cut' rating.
    Allowed values are 'EX', 'IDEAL', or 'EXCELLENT'. For non-Round shapes, return True.
    """
    shape = str(row.get('FinalShape', '')).upper()
    if shape == 'ROUND':
        cut = str(row.get('cut', '')).strip().upper()
        return cut in ['EX', 'IDEAL', 'EXCELLENT']
    else:
        return True

def clarity_group(clarity_raw):
    """Simplify clarity (e.g. "VVS2" or "VS2") to "VVS" or "VS"."""
    clarity_raw = str(clarity_raw).upper().strip()
    if clarity_raw.startswith("VVS"):
        return "VVS"
    elif clarity_raw.startswith("VS"):
        return "VS"
    else:
        return None

def clarity_matches(row_clarity, group_clarity):
    """Return True if the diamond's simplified clarity matches the desired group."""
    grp = clarity_group(row_clarity)
    if group_clarity == 'VS-VVS':
        return grp in ['VVS', 'VS']
    else:
        return grp == group_clarity

# --- Import & Normalize Raw Data ---

df = pd.read_csv(local_file, sep=',', low_memory=False,
                 dtype={'floCol': str, 'canadamarkeligible': str})

# Normalize column names (strip spaces and convert to lowercase)
df.columns = [col.strip().lower() for col in df.columns]
print("Normalized columns:", df.columns.tolist())

# Filter rows:
#   • lab must be "IGI" or "GIA" (using the correct column 'lab')
#   • col (color) must be one of D, E, or F
#   • Both image and video columns must be nonempty.
df = df[df['lab'].isin(['IGI', 'GIA'])]
df = df[df['col'].isin(['D', 'E', 'F'])]
df = df[df['image'].notnull() & (df['image'].astype(str).str.strip() != "")]
df = df[df['video'].notnull() & (df['video'].astype(str).str.strip() != "")]

# Map shapes and compute additional columns.
df['FinalShape'] = df.apply(map_shape, axis=1)
allowed_shapes = ['Round', 'Oval', 'Princess', 'Emerald', 'Asscher', 'Cushion', 'Marquise', 'Pear', 'Radiant', 'Heart']
df = df[df['FinalShape'].isin(allowed_shapes)]
df['Ratio'] = df.apply(compute_ratio, axis=1)
df['Measurement'] = df.apply(compute_measurement, axis=1)
df['v360 link'] = df['reportno'].apply(lambda x: f"https://loupe360.com/diamond/{x}/video/500/500")

# Apply quality filters: for polish and symmetry (for all shapes) and cut (only for Round).
df = df[df.apply(valid_cut, axis=1)]
df = df[df['pol'].astype(str).str.strip().str.upper().isin(['EX', 'EXCELLENT'])]
df = df[df['symm'].astype(str).str.strip().str.upper().isin(['EX', 'EXCELLENT'])]

# --- Balanced Selection by Shape & Carat/Clarity Groups ---

groups = [
    {'min_carat': 0.95, 'max_carat': 1.10, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 0.95, 'max_carat': 1.10, 'clarity': 'VS', 'count': 20},
    {'min_carat': 1.45, 'max_carat': 1.60, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 1.45, 'max_carat': 1.60, 'clarity': 'VS', 'count': 20},
    {'min_carat': 1.95, 'max_carat': 2.10, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 1.95, 'max_carat': 2.10, 'clarity': 'VS', 'count': 20},
    {'min_carat': 2.45, 'max_carat': 2.60, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 2.45, 'max_carat': 2.60, 'clarity': 'VS', 'count': 20},
    {'min_carat': 2.95, 'max_carat': 3.10, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 2.95, 'max_carat': 3.10, 'clarity': 'VS', 'count': 20},
    {'min_carat': 3.45, 'max_carat': 3.60, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 3.45, 'max_carat': 3.60, 'clarity': 'VS', 'count': 20},
    {'min_carat': 3.95, 'max_carat': 4.10, 'clarity': 'VVS', 'count': 28},
    {'min_carat': 3.95, 'max_carat': 4.10, 'clarity': 'VS', 'count': 20},
    {'min_carat': 4.50, 'max_carat': 4.99, 'clarity': 'VS-VVS', 'count': 28},
    {'min_carat': 5.00, 'max_carat': 5.99, 'clarity': 'VS-VVS', 'count': 28},
    {'min_carat': 6.00, 'max_carat': 6.99, 'clarity': 'VS-VVS', 'count': 28},
    {'min_carat': 7.00, 'max_carat': 7.99, 'clarity': 'VS-VVS', 'count': 28},
    {'min_carat': 8.00, 'max_carat': 8.99, 'clarity': 'VS-VVS', 'count': 28},
]

target_per_shape = 480
final_selection = []

for shape in allowed_shapes:
    shape_pool = df[df['FinalShape'] == shape]
    shape_selected = pd.DataFrame()
    for grp in groups:
        group_df = shape_pool[
            (shape_pool['carats'] >= grp['min_carat']) &
            (shape_pool['carats'] <= grp['max_carat']) &
            (shape_pool['clar'].apply(lambda x: clarity_matches(x, grp['clarity'])))
        ]
        group_df_sorted = group_df.sort_values(by='price', ascending=True)
        group_sel = group_df_sorted.head(grp['count'])
        shape_selected = pd.concat([shape_selected, group_sel])
    shape_selected = shape_selected.drop_duplicates()
    current_count = len(shape_selected)
    if current_count < target_per_shape:
        additional_candidates = shape_pool.drop(shape_selected.index, errors='ignore')
        additional_sorted = additional_candidates.sort_values(by='price', ascending=True)
        needed = target_per_shape - current_count
        additional_sel = additional_sorted.head(needed)
        shape_selected = pd.concat([shape_selected, additional_sel])
    if len(shape_selected) > target_per_shape:
        shape_selected = shape_selected.sort_values(by='price', ascending=True).head(target_per_shape)
    final_selection.append(shape_selected)

final_df = pd.concat(final_selection).reset_index(drop=True)
print(f"Balanced selection complete: {len(final_df)} diamonds selected.")

# Add a stock id column (e.g., "NVL-YYYYMMDD-01")
today_str = datetime.today().strftime("%Y%m%d")
final_df['stock id'] = final_df.index + 1
final_df['stock id'] = final_df['stock id'].apply(lambda x: f"NVL-{today_str}-{x:02d}")

# Rename columns for Shopify formatting.
final_df.rename(columns={
    'lab': 'LAB',  # Using the 'lab' column as it exists in your CSV.
    'reportno': 'REPORT NO',
    'FinalShape': 'Shape',
    'carats': 'Carat',
    'col': 'Color',
    'clar': 'Clarity',
    'price': 'Price',
    'cut': 'Cut',
    'pol': 'Polish',
    'symm': 'Symmetry',
    'flo': 'Fluor'
}, inplace=True)

# Write the transformed diamonds file using a relative path.
selected_output_filename = "transformed_diamonds.csv"
final_df.to_csv(selected_output_filename, index=False)
print(f"Selected diamonds file written with {len(final_df)} diamonds at {selected_output_filename}.")

##############################################
# PART 2: PRICE CONVERSION & SHOPIFY UPLOAD FORMAT TRANSFORMATION
##############################################

def get_usd_to_cad_rate():
    url = "https://v6.exchangerate-api.com/v6/20155ba28afe7c763416cc23/latest/USD"
    try:
        response = requests.get(url)
        data = response.json()
        return data["conversion_rates"]["CAD"]
    except Exception as e:
        print("Error fetching exchange rate:", e)
        return 1.0  # fallback

usd_to_cad_rate = get_usd_to_cad_rate()
print(f"USD to CAD rate: {usd_to_cad_rate}")

def markup(x):
    cad = x * usd_to_cad_rate
    base = cad * 1.05 * 1.13
    additional = (
        250 if cad <= 500 else
        405 if cad <= 1000 else
        600 if cad <= 1500 else
        800 if cad <= 2000 else
        1000 if cad <= 2500 else
        1200 if cad <= 3000 else
        1300 if cad <= 5000 else
        1600 if cad <= 100000 else
        0
    ) * 1.15
    return round(base + additional, 2)

final_df['CAD_Price'] = final_df['Price'].apply(markup).round(2)
final_df['Compare_At_Price'] = (final_df['CAD_Price'] * 1.5).round(2)
final_df['Ratio'] = final_df['Ratio'].round(2)

custom_collection = f"Lab-Created Diamonds-{today_str}"

def clean_image_url(url):
    if pd.isna(url):
        return url
    if "?" in url:
        return url.split("?")[0]
    return url

def generate_handle(row):
    return f"Lab-Grown-{row['Shape']}-Diamond-{row['Carat']}-Carat-{row['Color']}-{row['Clarity']}-Clarity-{row['REPORT NO']}"

def generate_title(row):
    return f"{row['Shape']}-{row['Carat']}-Carat-{row['Color']}-{row['Clarity']}-{row['LAB']}-Certified - {row['REPORT NO']}"

def generate_body_html(row):
    return (f"Discover timeless beauty with our {row['Shape']} Cut Diamond, a stunning {row['Carat']}-carat gem "
            f"boasting a rare {row['Color']} color and impeccable {row['Clarity']} clarity. All our diamonds feature "
            f"the best cut, polish, and symmetry ratios and are certified by {row['LAB']}. Report Number: {row['REPORT NO']}. "
            f"Elevate your jewelry collection with this exquisite combination of elegance and brilliance. Explore now for a truly exceptional and radiant choice.")

def generate_tags(row):
    return f"Lab-Created Diamonds-{today_str}"

def generate_image_alt(row):
    return (f"Lab-Grown {row['Shape']} Diamond - {row['Carat']} Carats, {row['Color']} Color, {row['Clarity']} Clarity - "
            f"Certified by {row['LAB']} - Report Number: {row['REPORT NO']}")

def generate_title_tag(row):
    return (f"Lab-Grown {row['Shape']} Diamond, {row['Carat']} Carats, {row['Color']} Color, {row['Clarity']} Clarity, "
            f"{row['LAB']} Certified - Report Number: {row['REPORT NO']}")

def generate_viewcertilink(row):
    report_no = row["REPORT NO"]
    lab = row["LAB"].upper()
    if lab == "IGI":
        return f"https://www.igi.org/verify-your-report/?r={report_no}"
    elif lab == "GIA":
        return f"https://www.gia.edu/report-check?locale=en_US&reportno={report_no}"
    else:
        return ""

shopify_df = pd.DataFrame({
    "Handle": final_df.apply(generate_handle, axis=1),
    "Title": final_df.apply(generate_title, axis=1),
    "Body HTML": final_df.apply(generate_body_html, axis=1),
    "Tags": final_df.apply(generate_tags, axis=1),
    "Image Src": final_df["image"].apply(clean_image_url),
    "Image Alt Text": final_df.apply(generate_image_alt, axis=1),
    "Variant Price": final_df["CAD_Price"].apply(lambda x: f"${x:.2f}"),
    "Variant Compare At Price": final_df["Compare_At_Price"].apply(lambda x: f"${x:.2f}"),
    "Metafield: title_tag [string]": final_df.apply(generate_title_tag, axis=1),
    "Metafield: description_tag [string]": final_df.apply(generate_body_html, axis=1),
    "Metafield: custom.diacertilab [single_line_text_field]": final_df["LAB"],
    "Metafield: custom.diacertino [number_integer]": final_df["REPORT NO"],
    "Metafield: custom.shape [single_line_text_field]": final_df["Shape"],
    "Metafield: custom.diacarat [number_decimal]": final_df["Carat"],
    "Metafield: custom.diacolor [single_line_text_field]": final_df["Color"],
    "Metafield: custom.diaclarity [single_line_text_field]": final_df["Clarity"],
    "Metafield: custom.diacut [single_line_text_field]": final_df["Cut"],
    "Metafield: custom.diapolish [single_line_text_field]": final_df["Polish"],
    "Metafield: custom.diasymmetry [single_line_text_field]": final_df["Symmetry"],
    "Metafield: custom.diaflourence [single_line_text_field]": final_df["Fluor"],
    "Metafield: custom.360_video [url]": final_df["v360 link"],
    "Metafield: custom.viewcertilink [url]": final_df.apply(generate_viewcertilink, axis=1),
    "Metafield: custom.diameasurement [single_line_text_field]": final_df["Measurement"],
    "Metafield: custom.diaratio [number_decimal]": final_df["Ratio"].apply(lambda x: f"{x:.2f}"),
    "Custom Collections": custom_collection,
    "Metafield: shopify.jewelry-type [list.metaobject_reference]": "shopify--jewelry-type.fine-jewelry",
    "Metafield: shopify.target-gender [list.metaobject_reference]": "shopify--target-gender.unisex",
    "Metafield: shopify.jewelry-material [list.metaobject_reference]": "shopify--jewelry-material.diamond",
    "Metafield: shopify.color-pattern [list.metaobject_reference]": "shopify--color-pattern.gold, shopify--color-pattern.white, shopify--color-pattern.rose-gold",
    "Variant Metafield: mm-google-shopping.age_group [single_line_text_field]": "adult",
    "Variant Metafield: mm-google-shopping.gender [single_line_text_field]": "unisex",
    "Variant Metafield: mm-google-shopping.color [single_line_text_field]": "white/yellow/rose gold",
    "Metafield: msft_bingads.bing_product_category [string]": "Apparel & Accessories > Jewelry > Loose Stones > Diamonds",
    "Metafield: msft_bingads.age_group [string]": "adult",
    "Metafield: msft_bingads.gender [string]": "unisex",
    "Vendor": "Lab-Grown",
    "Type": "Lab-Grown Diamond",
    "Template Suffix": "lab_grown-diamond",
    "Category: ID": "331",
    "Category: Name": "Jewelry",
    "Category": "Apparel & Accessories > Jewelry",
    "Variant Taxable": "FALSE",
    "Included / Canada": "TRUE",
    "Included / International": "TRUE",
    "Included / United States": "TRUE"
})

# Use a relative path and today's date for the final Shopify CSV file.
shopify_output_filename = f"shopify-lg-main-{today_str}.csv"
shopify_df.to_csv(shopify_output_filename, index=False)
print(f"Shopify upload file created with {len(shopify_df)} diamonds at {shopify_output_filename}.")

##############################################
# PART 3: UPLOAD TO GOOGLE CLOUD STORAGE
##############################################

def upload_to_gcs(source_file, destination_blob, bucket_name):
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    :param source_file: Path to the local file.
    :param destination_blob: Destination path (including folders) in the bucket.
    :param bucket_name: Name of the bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    print(f"File {source_file} uploaded to {destination_blob} in bucket {bucket_name}.")

# The bucket name and destination path (folder structure in the bucket)
bucket_name = "sitemaps.leeladiamond.com"
destination_blob = f"shopify final/{shopify_output_filename}"
upload_to_gcs(shopify_output_filename, destination_blob, bucket_name)
