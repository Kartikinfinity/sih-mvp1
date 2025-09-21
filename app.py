"""
Space Debris MVP - Streamlit (full end-to-end demo)

Features:
 - Loads TLEs (CelesTrak gp.php) or uses bundled sample TLEs (offline).
 - Uses Skyfield to propagate objects and estimate position + velocity.
 - 3D globe visualization with Plotly (orbit traces + current positions).
 - Conjunction scan (closest-approach) with threshold (default 10 km).
 - Visual collision alerts + connector lines.
 - Simulated avoidance maneuver (green trajectory show).
 - Debris analysis (size, material, weight, altitude, velocity) persisted in SQLite.
 - Data sharing: export CSV and "share" simulation (records in DB).
 - Recycling suggestions: cluster by material + altitude bins, draw mission path.

How to run:
 1) python3 -m venv venv
 2) source venv/bin/activate   # Windows: venv\\Scripts\\activate
 3) pip install -r requirements.txt
 4) streamlit run app.py
"""

from pathlib import Path
import os, math, time, sqlite3, io
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from skyfield.api import load, EarthSatellite

# ---------- Configuration ----------
EARTH_RADIUS_KM = 6371.0
TS = load.timescale()
DB_FILE = "debris_demo.db"
MAX_PLOT_OBJECTS = 40  # keep plots responsive
SAMPLE_TLES = """ISS (ZARYA)
1 25544U 98067A   25102.34784722  .00016717  00000-0  10249-3 0  9997
2 25544  51.6436 218.4283 0006465 233.7300 126.4279 15.49361166203342
HST
1 20580U 90037B   25102.79676481  .00000682  00000-0  21418-4 0  9991
2 20580  28.4692  34.9876 0002806  82.4321 277.7618 15.09245004639137
DEBRIS_1
1 43000U 99000A   25102.12345678  .00000000  00000-0  00000-0 0  9991
2 43000  98.0000 100.0000 0012345  45.0000 315.0000 14.00000000
DEBRIS_2
1 43001U 99001A   25102.22345678  .00000000  00000-0  00000-0 0  9992
2 43001  97.5000 150.0000 0012345  10.0000 200.0000 14.10000000
"""

# ---------- Utils: TLE download & parsing ----------
def download_celestrak(group='visual', filename='tles_download.tle'):
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=TLE"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with open(filename, 'w') as f:
        f.write(r.text)
    return filename

def parse_tle_text(text):
    lines = [L.rstrip() for L in text.splitlines() if L.strip()!='']
    sats = []
    i = 0
    while i < len(lines)-1:
        if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
            line1 = lines[i]; line2 = lines[i+1]
            name = "TLE_" + line1.split()[1]
            i += 2
        elif (i+2 < len(lines)) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            name = lines[i]; line1 = lines[i+1]; line2 = lines[i+2]
            i += 3
        else:
            i += 1; continue
        try:
            sats.append(EarthSatellite(line1, line2, name, TS))
        except Exception:
            continue
    return sats

def load_sats_from_file(fn):
    with open(fn,'r') as f:
        return parse_tle_text(f.read())

# ---------- Utils: geometry ----------
def latlonalt_from_sat_at(sat, t):
    sp = sat.at(t).subpoint()
    return float(sp.latitude.degrees), float(sp.longitude.degrees), float(sp.elevation.km)

def lla_to_ecef(lat_deg, lon_deg, alt_km):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return x, y, z

def sample_orbit_points(sat, center_time_utc, seconds_span=3600, step_seconds=30):
    seconds = np.arange(-seconds_span/2, seconds_span/2 + 1, step_seconds)
    times = TS.utc(center_time_utc.year, center_time_utc.month, center_time_utc.day,
                   center_time_utc.hour, center_time_utc.minute, center_time_utc.second + seconds)
    xs, ys, zs = [], [], []
    for t in times:
        try:
            lat, lon, alt = latlonalt_from_sat_at(sat, t)
            x,y,z = lla_to_ecef(lat, lon, alt)
            xs.append(x); ys.append(y); zs.append(z)
        except Exception:
            xs.append(np.nan); ys.append(np.nan); zs.append(np.nan)
    return np.array(xs), np.array(ys), np.array(zs), times

# ---------------- Image-analysis MVP helpers ----------------
import cv2
import base64
from PIL import Image
import io
import hashlib

# Try to import ultralytics (optional - better detection). If unavailable, the code falls back.
USE_YOLO = False
try:
    from ultralytics import YOLO
    # you can download/point to a small model like "yolov8n.pt" automatically if available
    try:
        ymodel = YOLO("yolov8n.pt")  # small model — place in repo or let user install
        USE_YOLO = True
    except Exception:
        # If model file missing, will still keep USE_YOLO False and fallback
        USE_YOLO = False
except Exception:
    USE_YOLO = False

def pil_to_cv2(im_pil):
    return cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(im_cv2):
    return Image.fromarray(cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB))

def read_imagefile_to_pil(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image

def run_yolo_detect(pil_image):
    """Return list of detections: {x1,y1,x2,y2,conf,cls}. Coordinates in pixels."""
    results = ymodel.predict(source=np.asarray(pil_image), imgsz=640, conf=0.25, max_det=10)
    # take first batch result
    dets = []
    if len(results) > 0:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else -1
                dets.append({'x1': int(xyxy[0]), 'y1': int(xyxy[1]), 'x2': int(xyxy[2]), 'y2': int(xyxy[3]), 'conf': conf, 'cls': cls})
    return dets

def contour_fallback_detect(pil_image):
    """Simple detection fallback: convert to gray, threshold, find contours; returns detections in pixels."""
    im = pil_to_cv2(pil_image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to detect bright objects
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = im.shape[:2]
    dets = []
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        if area < (w*h)*0.0005:  # skip tiny blobs (tweakable)
            continue
        dets.append({'x1': x, 'y1': y, 'x2': x+ww, 'y2': y+hh, 'conf': 0.5, 'cls': -1})
    # if nothing found, fallback to full-frame
    if len(dets) == 0:
        dets.append({'x1': 0, 'y1': 0, 'x2': w-1, 'y2': h-1, 'conf': 0.1, 'cls': -1})
    return dets

def estimate_material_from_patch(pil_image, bbox):
    """Heuristic color/reflectivity-based material estimate from bbox (returns label + score)."""
    im = pil_to_cv2(pil_image)
    x1,y1,x2,y2 = bbox
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(im.shape[1]-1,int(x2)), min(im.shape[0]-1,int(y2))
    patch = im[y1:y2, x1:x2]
    if patch.size == 0:
        return "Unknown", 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[:,:,2].mean()) / 255.0  # brightness
    s_mean = float(hsv[:,:,1].mean()) / 255.0  # saturation
    # heuristics:
    if v_mean > 0.7 and s_mean < 0.25:
        return "Aluminum-like (high reflectivity)", v_mean
    if s_mean > 0.35 and v_mean > 0.4:
        return "Solar-Cell-like (colored/reflective)", s_mean
    if v_mean < 0.35:
        return "Dark/Carbon-like", 1.0 - v_mean
    return "Mixed/Unknown", 0.5

def estimate_shape_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    w = max(1, x2-x1); h = max(1, y2-y1)
    ar = w / h
    if ar > 1.8:
        return "Long / Rod-like"
    if ar < 0.6:
        return "Tall / Cylinder-like"
    # near square
    return "Compact / Plate-like or Irregular"

def estimate_size_meters_from_bbox(bbox, image_width_px, fov_deg=10.0, distance_km=0.5):
    """
    Estimate physical size (meters) from bounding box angular size.
    - fov_deg: camera horizontal FOV in degrees (user input)
    - image_width_px: width of image in pixels
    - distance_km: distance to object (km) (user input or TLE-based)
    Returns: estimated_size_meters (float)
    """
    x1,y1,x2,y2 = bbox
    w_px = abs(x2 - x1)
    if image_width_px <= 0 or w_px <= 0:
        return float('nan')
    # angular width (deg) approximated by (w_px / image_width_px) * fov_deg
    theta_deg = (w_px / float(image_width_px)) * float(fov_deg)
    theta_rad = math.radians(theta_deg)
    # small-angle approximation: size ≈ distance_m * theta_rad
    distance_m = float(distance_km) * 1000.0
    size_m = distance_m * theta_rad
    return float(size_m)

def estimate_speed_from_two_images(bbox1, bbox2, dt_seconds, image_width_px, fov_deg, distance_km):
    """
    Estimate tangential speed (m/s) from pixel shift between two images.
    - bbox1/bbox2 center pixel positions
    - dt_seconds between captures
    """
    if dt_seconds <= 0:
        return float('nan')
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    dx_px = cx2 - cx1
    # convert pixel shift to radians using FOV horizontal
    ang_shift_deg = (dx_px / float(image_width_px)) * float(fov_deg)
    ang_shift_rad = math.radians(ang_shift_deg)
    distance_m = float(distance_km) * 1000.0
    arc_m = distance_m * ang_shift_rad
    return abs(arc_m / dt_seconds)  # m/s

def save_analysis_to_db(norad, name, material, size_label, type_label, weight_kg, alt_km):
    """Wrap up: uses your existing upsert_debris_record to store an analyzed object.
       norad can be None for synthetic items. name used as unique-ish key if norad missing."""
    try:
        if norad:
            upsert_debris_record(norad, name, material, size_label, type_label, weight_kg, alt_km)
        else:
            # If norad missing, try to create a synthetic key by hashing name
            fake_n = int(abs(int(hashlib.md5(name.encode()).hexdigest(),16) % (10**8)))
            upsert_debris_record(fake_n, name, material, size_label, type_label, weight_kg, alt_km)
    except Exception:
        pass

# ---------- Conjunction (closest approach) ----------
def compute_closest_approach(target_sat, other_sat, t0, span_seconds=3600, step_seconds=10):
    seconds = np.arange(-span_seconds/2, span_seconds/2 + 1, step_seconds)
    times = TS.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + seconds)
    try:
        pos1 = target_sat.at(times).position.km
        pos2 = other_sat.at(times).position.km
    except Exception:
        return float('inf'), None, None
    diffs = pos2 - pos1
    dists = np.sqrt(np.sum(diffs**2, axis=0))
    idx = int(np.nanargmin(dists))
    return float(dists[idx]), times[idx].utc_datetime(), {"idx": idx}

# ---------- Debris attribute heuristics ----------
def estimate_debris_attributes(sat):
    name = sat.name.upper() if hasattr(sat,'name') else "OBJ"
    key = getattr(sat, 'model', None)
    satnum = getattr(key, 'satnum', None) or hash(name)
    seed = int(abs(satnum)) % 100
    sizes = ['Tiny (<1m)', 'Small (1-10m)', 'Medium (10-100m)', 'Large (>100m)']
    materials = ['Aluminum', 'Titanium', 'Composites', 'SolarCells', 'Mixed']
    types = ['Fragment', 'Rocket Stage', 'Deployed Item', 'Paint Flake', 'Unknown']
    weight_kg = [1, 5, 20, 50, 120, 300, 800]
    size = sizes[seed % len(sizes)]
    material = materials[(seed//3) % len(materials)]
    type_ = types[(seed//5) % len(types)]
    weight = weight_kg[seed % len(weight_kg)]
    return size, material, type_, weight

# ---------- Database (SQLite) ----------
def init_db(path=DB_FILE):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''
      CREATE TABLE IF NOT EXISTS debris (
        norad INTEGER PRIMARY KEY,
        name TEXT,
        material TEXT,
        size TEXT,
        type TEXT,
        weight_kg REAL,
        alt_km REAL,
        last_seen TEXT
      )
    ''')
    cur.execute('''
      CREATE TABLE IF NOT EXISTS shared (id INTEGER PRIMARY KEY AUTOINCREMENT, norad INTEGER, org TEXT, at TEXT)
    ''')
    conn.commit(); conn.close()

def upsert_debris_record(norad, name, material, size, type_, weight, alt_km):
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute('''
      INSERT INTO debris (norad,name,material,size,type,weight_kg,alt_km,last_seen)
      VALUES (?,?,?,?,?,?,?,?)
      ON CONFLICT(norad) DO UPDATE SET
        name=excluded.name, material=excluded.material, size=excluded.size,
        type=excluded.type, weight_kg=excluded.weight_kg, alt_km=excluded.alt_km, last_seen=excluded.last_seen
    ''', (norad,name,material,size,type_,weight,alt_km,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def read_all_debris():
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute('SELECT norad,name,material,size,type,weight_kg,alt_km,last_seen FROM debris')
    rows = cur.fetchall(); conn.close()
    df = pd.DataFrame(rows, columns=['norad','name','material','size','type','weight_kg','alt_km','last_seen'])
    return df

def share_record(norad, org):
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute('INSERT INTO shared (norad,org,at) VALUES (?,?,?)', (norad, org, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

# ---------- Plot helpers ----------
def make_earth_points(res=30):
    u = np.linspace(0, 2*np.pi, res)
    v = np.linspace(0, np.pi, max(3, res//2))
    xs, ys, zs = [], [], []
    for uu in u:
        for vv in v:
            x = EARTH_RADIUS_KM * np.cos(uu) * np.sin(vv)
            y = EARTH_RADIUS_KM * np.sin(uu) * np.sin(vv)
            z = EARTH_RADIUS_KM * np.cos(vv)
            xs.append(x); ys.append(y); zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)

def build_plotly_figure(objects_to_plot, orbit_traces, conn_lines=None, maneuver_traces=None):
    fig = go.Figure()
    # Earth points (low-res)
    ex,ey,ez = make_earth_points(36)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='markers', marker=dict(size=1,color='rgb(10,80,120)'), name='Earth'))
    # orbit traces
    for t in orbit_traces:
        fig.add_trace(go.Scatter3d(x=t['x'], y=t['y'], z=t['z'], mode='lines', line=dict(width=2), name=t.get('name','orbit')))
    # objects markers
    for obj in objects_to_plot:
        fig.add_trace(go.Scatter3d(x=[obj['x']], y=[obj['y']], z=[obj['z']], mode='markers+text',
                                   marker=dict(size=obj.get('size_marker',4), color=obj.get('color','red')),
                                   name=obj['name'], text=[obj['name']], textposition='top center', hoverinfo='text', hovertext=obj.get('hover','')))
    # connection lines (e.g., conjunction red lines)
    if conn_lines:
        for c in conn_lines:
            fig.add_trace(go.Scatter3d(x=c['x'], y=c['y'], z=c['z'], mode='lines', line=dict(color='red', width=4,dash='dash'), name=c.get('name','conn')))
    # maneuver traces (green)
    if maneuver_traces:
        for m in maneuver_traces:
            fig.add_trace(go.Scatter3d(x=m['x'], y=m['y'], z=m['z'], mode='lines', line=dict(color='green', width=4), name=m.get('name','maneuver')))
    fig.update_layout(scene=dict(xaxis=dict(showgrid=False,visible=False), yaxis=dict(showgrid=False,visible=False), zaxis=dict(showgrid=False,visible=False)),
                      showlegend=True, margin=dict(l=0,r=0,t=0,b=0), height=720)
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Space Debris MVP")
st.title("ASTRALINK:- SPACE DEBRIS DETECTION AND ANALYSIS")

init_db()  # ensure DB exists

# Sidebar: data source + controls
with st.sidebar:
    st.header("Data & Scan Controls")
    use_live = st.checkbox("Fetch live TLEs from CelesTrak (internet required)", value=False)
    group = st.selectbox("CelesTrak GROUP", ['visual','active','stations','cubesat','debris'], index=0)
    if st.button("Load / Refresh catalog"):
        try:
            if use_live:
                fn = download_celestrak(group=group, filename='tles_download.tle')
                sats = load_sats_from_file(fn)
            else:
                sats = parse_tle_text(SAMPLE_TLES)
            st.session_state['sats_list'] = sats
            st.success(f"Loaded {len(sats)} objects.")
        except Exception as e:
            st.warning("Failed to fetch. Using bundled sample.")
            st.session_state['sats_list'] = parse_tle_text(SAMPLE_TLES)
    # default load on first run
    if 'sats_list' not in st.session_state:
        st.session_state['sats_list'] = parse_tle_text(SAMPLE_TLES)

    threshold_km = st.number_input("Collision threshold (km)", min_value=0.1, value=10.0, step=0.1)
    scan_minutes = st.slider("Scan window (minutes)", 1, 180, 30)
    step_sec = st.selectbox("Scan step (sec)", [1,2,5,10,30,60], index=3)
    dv_m_s = st.number_input("Maneuver delta-v (m/s) (demo)", min_value=0.1, value=0.8, step=0.1)
    bin_km = st.number_input("Altitude bin for recycling (km)", min_value=10, value=100, step=10)
    st.markdown("---")
    st.markdown("Demo helpers:")
    if st.button("Create demo close approach (force)"):
        st.session_state['force_close'] = True
        st.experimental_rerun()
    st.markdown("Notes: for production you must replace simulated heuristics with sensor RCS/spectra & covariance-based conjunction math.")

# Layout: left globe, right details
col1, col2 = st.columns([3,1])

# Prepare objects: satellites from session
sats = st.session_state.get('sats_list', parse_tle_text(SAMPLE_TLES))
now = datetime.utcnow().replace(tzinfo=timezone.utc)

# Populate analysis table and write to DB
analysis_rows = []
for s in sats:
    try:
        lat, lon, alt = latlonalt_from_sat_at(s, TS.utc(now.year, now.month, now.day, now.hour, now.minute, now.second))
    except Exception:
        lat = lon = 0.0; alt = 0.0
    size, material, type_, weight = estimate_debris_attributes(s)
    norad = getattr(s.model, 'satnum', None)
    analysis_rows.append({'norad': norad, 'name': s.name, 'size': size, 'material': material, 'type': type_, 'weight_kg': weight, 'alt_km': alt})
    # upsert to DB
    if norad:
        upsert_debris_record(norad, s.name, material, size, type_, weight, alt)
analysis_df = pd.DataFrame(analysis_rows)

# Right column: data + sharing + recycling
with col2:
    st.subheader("Debris catalog & actions")
    st.write("Database-backed catalog (simulated attributes).")
    df_db = read_all_debris()
    st.dataframe(df_db, use_container_width=True, height=480)

    st.markdown("### Share / Export")
    org = st.text_input("Organization name to 'share' selected NORAD with (demo):")
    norad_choices = df_db['norad'].dropna().astype(int).astype(str).tolist()
    sel_norad = st.selectbox("Select NORAD to share", options=['']+norad_choices)
    if st.button("Share selected record (demo)"):
        if sel_norad and org:
            share_record(int(sel_norad), org)
            st.success(f"Shared NORAD {sel_norad} with {org} (recorded in DB).")
        else:
            st.warning("Choose a NORAD and organization name first.")
    st.download_button("Export catalog CSV", df_db.to_csv(index=False).encode('utf-8'), file_name='debris_catalog.csv')

    st.markdown("---")
    st.subheader("Reuse / Recycling")
    st.write("Group debris by material + altitude bins and suggest a collection mission (demo).")
    if st.button("Compute recycling suggestions"):
        if df_db.empty:
            st.info("No debris records found.")
        else:
            df_db['alt_bin'] = (df_db['alt_km'] / bin_km).round().astype(int)
            groups = df_db.groupby(['material','alt_bin'])
            suggestions = []
            for (mat,binidx),g in groups:
                if len(g) >= 2:
                    suggestions.append({'material': mat, 'alt_bin': int(binidx), 'count': len(g), 'examples': list(g['name'].head(6))})
            if suggestions:
                sug_df = pd.DataFrame(suggestions).sort_values('count', ascending=False)
                st.table(sug_df)
                st.session_state['recycling_suggestions'] = suggestions
            else:
                st.info("No strong clusters found (try lowering bin size or load larger catalog).")
    if st.button("Show top recycling mission on globe"):
     st.markdown("Image-based Debris Analysis")
st.write("Upload a debris image (or two) and set camera/distance info. The app will detect object(s), estimate size, shape, material, and optionally speed.")

with st.expander("Run image analysis"):
    uploaded = st.file_uploader("Upload debris image (single) — clear lighting/contrast helps", type=['jpg','png','jpeg'])
    uploaded2 = st.file_uploader("Optional: second image for speed estimate (same object) — provide time gap", type=['jpg','png','jpeg'])
    sample_demo = st.button("Use sample demo image (bundled)")

    fov_deg = st.number_input("Camera horizontal FOV (degrees)", min_value=0.1, max_value=120.0, value=10.0, step=0.1)
    distance_km = st.number_input("Estimated distance to object (km)", min_value=0.001, max_value=2000.0, value=0.5, step=0.01)
    time_gap_s = st.number_input("If supplying 2 images: time difference between them (seconds)", min_value=0.1, value=1.0, step=0.1)
    norad_for_save = st.text_input("Optional: NORAD id to tag this analysis (leave blank for synthetic)", value='')

    run_analysis = st.button("Run analysis")

    # load sample if requested
    if sample_demo and 'sample_demo_loaded' not in st.session_state:
        try:
            # Try to load bundled sample: you can add one to your repo named 'sample_debris.jpg'
            sample_path = Path("sample_debris.jpg")
            if sample_path.exists():
                st.session_state['sample_image'] = sample_path.read_bytes()
            else:
                # fallback: create synthetic blank small image
                im = Image.new('RGB', (800,600), color=(50,50,60))
                st.session_state['sample_image'] = io.BytesIO()
                im.save(st.session_state['sample_image'], format='JPEG')
                st.session_state['sample_image'] = st.session_state['sample_image'].getvalue()
        except Exception:
            st.warning("Could not load sample image.")
        st.experimental_rerun()

    # pick image source
    pil_img = None
    pil_img2 = None
    if uploaded is not None:
        pil_img = read_imagefile_to_pil(uploaded)
    elif 'sample_image' in st.session_state:
        try:
            pil_img = Image.open(io.BytesIO(st.session_state['sample_image'])).convert("RGB")
        except Exception:
            pil_img = None

    if uploaded2 is not None:
        pil_img2 = read_imagefile_to_pil(uploaded2)
    elif pil_img is not None and uploaded2 is None:
        pil_img2 = None

    if run_analysis and pil_img is None:
        st.warning("Please provide an image first.")
    if run_analysis and pil_img is not None:
        # detect objects
        h,w = pil_img.size[1], pil_img.size[0]  # note PIL size is (w,h) => careful
        if USE_YOLO:
            try:
                dets = run_yolo_detect(pil_img)
            except Exception:
                dets = contour_fallback_detect(pil_img)
        else:
            dets = contour_fallback_detect(pil_img)

        # pick best detection (highest conf)
        dets_sorted = sorted(dets, key=lambda x: x.get('conf',0.0), reverse=True)
        best = dets_sorted[0]
        bbox = (best['x1'], best['y1'], best['x2'], best['y2'])

        # estimate attributes
        material_label, material_score = estimate_material_from_patch(pil_img, bbox)
        shape_label = estimate_shape_from_bbox(bbox)
        size_m = estimate_size_meters_from_bbox(bbox, image_width_px=w, fov_deg=fov_deg, distance_km=distance_km)
        # speed estimate if second image provided
        speed_m_s = None
        if pil_img2 is not None:
            # detect second image object
            if USE_YOLO:
                try:
                    dets2 = run_yolo_detect(pil_img2)
                except Exception:
                    dets2 = contour_fallback_detect(pil_img2)
            else:
                dets2 = contour_fallback_detect(pil_img2)
            if len(dets2) > 0:
                best2 = sorted(dets2, key=lambda x: x.get('conf',0.0), reverse=True)[0]
                bbox2 = (best2['x1'], best2['y1'], best2['x2'], best2['y2'])
                speed_m_s = estimate_speed_from_two_images(bbox, bbox2, time_gap_s, image_width_px=w, fov_deg=fov_deg, distance_km=distance_km)

        # show results and annotated image
        im_cv = pil_to_cv2(pil_img)
        cv2.rectangle(im_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        label_txt = f"Size≈{size_m:.2f}m | {material_label} | {shape_label}"
        cv2.putText(im_cv, label_txt, (max(5,bbox[0]), max(20,bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        st.image(cv2_to_pil(im_cv), caption="Annotated detection", use_column_width=True)

        st.markdown("**Analysis output**")
        st.write({
            "estimated_size_m": round(float(size_m),3),
            "material": material_label,
            "shape": shape_label,
            "distance_km_used": distance_km,
            "camera_fov_deg": fov_deg,
            "speed_m_s (from 2 images)": (round(float(speed_m_s),3) if speed_m_s is not None else "N/A"),
            "detection_confidence": best.get('conf', None)
        })

        # save to DB
        save_name = f"IMG_ANALYSIS_{int(time.time())}"
        try:
            norad_val = int(norad_for_save) if norad_for_save.strip()!='' else None
        except Exception:
            norad_val = None
        save_analysis_to_db(norad_val, save_name, material_label, shape_label, "ImageAnalysis", weight_kg=size_m*0.0 + 1.0, alt_km=distance_km*1000.0/1000.0)  # alt_km roughly = distance_km
        st.success("Analysis saved (catalog). You can view it in the Debris catalog above.")

# Left: main globe + controls
with col1:
    st.subheader("Interactive Globe — Orbits & Alerts")
    # choose target
    name_to_sat = {s.name: s for s in sats}
    target_name = st.selectbox("Pick a spacecraft to monitor (primary target)", options=list(name_to_sat.keys()), index=0)
    target_sat = name_to_sat[target_name]

    # Prepare orbit traces & markers (limit to MAX_PLOT_OBJECTS)
    sample_list = [target_sat] + [s for s in sats if s != target_sat][:MAX_PLOT_OBJECTS-1]
    orbit_traces = []
    plot_markers = []
    for s in sample_list:
        ox,oy,oz,times = sample_orbit_points(s, now, seconds_span=3600, step_seconds=30)
        orbit_traces.append({'name': s.name, 'x': ox, 'y': oy, 'z': oz})
        try:
            lat, lon, alt = latlonalt_from_sat_at(s, TS.utc(now.year, now.month, now.day, now.hour, now.minute, now.second))
            x,y,z = lla_to_ecef(lat, lon, alt)
        except Exception:
            x=y=z=0.0
        color = 'blue' if s == target_sat else 'red'
        hover = f"{s.name}<br>alt(km): {alt:.1f}"
        plot_markers.append({'name': s.name, 'x': x, 'y': y, 'z': z, 'color': color, 'size_marker': 5, 'hover': hover})

    # add any synthetic forced close object if requested
    if st.session_state.get('force_close', False):
        # create synthetic debris near target (8 km offset)
        tnow = TS.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
        try:
            lat_t, lon_t, alt_t = latlonalt_from_sat_at(target_sat, tnow)
            synth_lat = lat_t + 0.05; synth_lon = lon_t + 0.05; synth_alt = alt_t + 8.0
            sx,sy,sz = lla_to_ecef(synth_lat, synth_lon, synth_alt)
            plot_markers.append({'name':'DEMO_DEBRIS_CLOSE','x':sx,'y':sy,'z':sz,'color':'orange','size_marker':7,'hover':'Demo debris (forced near miss)'})
            st.success("Demo close approach object added (visible on globe).")
        except Exception:
            st.warning("Could not create demo debris.")
        st.session_state['force_close'] = False

    # if recycling path requested, create connection trace
    conn_lines = []
    if st.session_state.get('show_recycle', False) and st.session_state.get('recycling_suggestions'):
        top = st.session_state['recycling_suggestions'][0]
        # find items matching top suggestion
        df_db = read_all_debris()
        matches = df_db[(df_db['material']==top['material']) & ( (df_db['alt_km']/bin_km).round().astype(int)==top['alt_bin'] )]
        coords = []
        for _,row in matches.head(6).iterrows():
            coords.append(lla_to_ecef(row['alt_km']*0, 0, 0))  # placeholder -> we will produce accurate coords below
        # Better: get real coords by matching sats list
        conn_x, conn_y, conn_z = [], [], []
        for name in top['examples']:
            s = next((ss for ss in sats if ss.name == name), None)
            if s:
                try:
                    lat, lon, alt = latlonalt_from_sat_at(s, TS.utc(now.year, now.month, now.day, now.hour, now.minute, now.second))
                    x,y,z = lla_to_ecef(lat, lon, alt); conn_x.append(x); conn_y.append(y); conn_z.append(z)
                except Exception:
                    continue
        if len(conn_x) >= 2:
            conn_lines.append({'x': conn_x, 'y': conn_y, 'z': conn_z, 'name': f"recycle_{top['material']}"})
        st.session_state['show_recycle'] = False

    # Conjunction scan - run on demand
    run_scan = st.button("Run collision scan")
    conn_lines_scan = []
    maneuver_traces = []
    if run_scan:
        results = []
        for s in sats:
            if s==target_sat: continue
            try:
                md, tmin, det = compute_closest_approach(target_sat, s, now, span_seconds=scan_minutes*60, step_seconds=step_sec)
                results.append({'name': s.name, 'norad': getattr(s.model,'satnum',None), 'min_dist_km': md, 'time_of_min': tmin})
            except Exception:
                continue
        resdf = pd.DataFrame(results).sort_values('min_dist_km')
        st.write("Top close approaches (km):")
        st.dataframe(resdf.head(20))
        close = resdf[resdf['min_dist_km'] <= threshold_km]
        if not close.empty:
            st.error(f"{len(close)} close approaches within threshold ({threshold_km} km).")
            # draw connectors & suggest maneuver for the worst one (smallest distance)
            worst = close.iloc[0]
            st.write("Closest:", worst['name'], f"{worst['min_dist_km']:.3f} km at {worst['time_of_min']}")
            # compute positions at time_of_min and draw red connector
            try:
                tmin = worst['time_of_min'].replace(tzinfo=timezone.utc)
                # ECI -> geodetic conversion for both sats at that time:
                tm = TS.utc(tmin.year, tmin.month, tmin.day, tmin.hour, tmin.minute, tmin.second)
                p_target = target_sat.at(tm).position.km
                p_other = next(s for s in sats if s.name==worst['name']).at(tm).position.km
                # Convert ECI to latlonalt for plotting; we can reuse sampling by computing geodetic subpoint:
                la_t, lo_t, al_t = latlonalt_from_sat_at(target_sat, tm)
                la_o, lo_o, al_o = latlonalt_from_sat_at(next(s for s in sats if s.name==worst['name']), tm)
                x1,y1,z1 = lla_to_ecef(la_t, lo_t, al_t); x2,y2,z2 = lla_to_ecef(la_o, lo_o, al_o)
                conn_lines_scan.append({'x':[x1,x2],'y':[y1,y2],'z':[z1,z2],'name':f"conn_{worst['name']}"})
                # Suggest avoidance: simple altitude raise of target proportional to dv_m_s (demo)
                delta_h = dv_m_s * 0.02  # demo mapping m/s -> km
                # build maneuver trace by sampling target orbit and pushing radius outward by delta_h
                ox,oy,oz,times = sample_orbit_points(target_sat, now, seconds_span=3600, step_seconds=30)
                man_x = []; man_y = []; man_z = []
                for xi,yi,zi in zip(ox,oy,oz):
                    vec = np.array([xi,yi,zi]); norm = np.linalg.norm(vec)
                    if norm==0:
                        man_x.append(xi); man_y.append(yi); man_z.append(zi); continue
                    new_norm = norm + delta_h
                    new_vec = vec * (new_norm/norm)
                    man_x.append(new_vec[0]); man_y.append(new_vec[1]); man_z.append(new_vec[2])
                maneuver_traces.append({'x': man_x, 'y': man_y, 'z': man_z, 'name': f"{target_sat.name}_maneuver"})
                st.success("Alert drawn and suggested green avoidance path added (demo).")
            except Exception as e:
                st.warning("Could not draw connectors/maneuver: " + str(e))
        else:
            st.success("No close approaches found in scanned window.")

    # Build & show figure
    fig = build_plotly_figure(plot_markers, orbit_traces, conn_lines + conn_lines_scan, maneuver_traces)
    st.plotly_chart(fig, use_container_width=True)

