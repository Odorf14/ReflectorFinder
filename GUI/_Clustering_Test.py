import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import tempfile

# Path to your CSV files
csv_folder = r'C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\GUI\_DXF\MissingReflectors'  # Change to your folder path
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))


# helper to read a single file (only required columns) and return coords or a skip reason
def read_coords(file_path):
    try:
        if os.path.getsize(file_path) == 0:
            return None, (file_path, 'empty file')
        # read only the needed columns to speed parsing
        df = pd.read_csv(file_path, usecols=['WorldX', 'WorldY'])
    except pd.errors.EmptyDataError:
        return None, (file_path, 'pandas EmptyDataError')
    except pd.errors.ParserError as e:
        return None, (file_path, f'ParserError: {e}')
    except ValueError:
        # raised when usecols are not present
        return None, (file_path, 'missing columns')
    except Exception as e:
        return None, (file_path, f'Other error: {e}')

    coords = df[['WorldX', 'WorldY']].dropna().values
    if coords.size == 0:
        return None, (file_path, 'no valid WorldX/WorldY rows')
    return coords, None


# Configuration: on Windows ProcessPoolExecutor requires the `if __name__ == '__main__'` guard
# to avoid spawn/import issues. Keep it disabled by default here; enable manually if you run
# this script as a module with the proper guard.
USE_PROCESS_POOL = False
USE_PYARROW = True
USE_MEMMAP = True  # when True, create a disk-backed memmap and fill it in chunks to avoid high RAM spikes
MEMMAP_PATH = os.path.join(tempfile.gettempdir(), f'coords_{os.getpid()}.memmap')
CHUNKSIZE = 100_000

# optional memory reporting
try:
    import psutil
    psutil_available = True
except Exception:
    psutil_available = False

def print_mem(msg=''):
    if psutil_available:
        proc = psutil.Process()
        rss = proc.memory_info().rss / (1024**2)
        print(f"MEM {msg}: {rss:.1f} MB")
    else:
        print(f"MEM {msg}: psutil not available")

def read_coords_wrapper(args):
    # wrapper to allow ProcessPoolExecutor which requires picklable callables
    fpath, use_pyarrow = args
    # inside processes we'll call read_coords but switch engine if requested
    try:
        if os.path.getsize(fpath) == 0:
            return None, (fpath, 'empty file')
        if use_pyarrow:
            try:
                df = pd.read_csv(fpath, usecols=['WorldX', 'WorldY'], engine='pyarrow')
            except Exception:
                df = pd.read_csv(fpath, usecols=['WorldX', 'WorldY'])
        else:
            df = pd.read_csv(fpath, usecols=['WorldX', 'WorldY'])
    except pd.errors.EmptyDataError:
        return None, (fpath, 'pandas EmptyDataError')
    except pd.errors.ParserError as e:
        return None, (fpath, f'ParserError: {e}')
    except ValueError:
        return None, (fpath, 'missing columns')
    except Exception as e:
        return None, (fpath, f'Other error: {e}')

    coords = df[['WorldX', 'WorldY']].dropna().values
    if coords.size == 0:
        return None, (fpath, 'no valid WorldX/WorldY rows')
    return coords, None


# Read files in parallel to reduce IO wait time
start_io = time.time()
points = None
skipped_files = []

if USE_MEMMAP:
    # two-pass approach: first count valid rows, then allocate memmap and fill in chunks
    print_mem('before count')
    total_rows = 0
    cols_lower = ['worldx', 'worldy', 'lgvx', 'lgvy']
    for fpath in csv_files:
        try:
            for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE):
                # normalize column names to lowercase for case-insensitive matching
                chunk.columns = [c.lower() for c in chunk.columns]
                # ensure columns exist in order and fill missing with NaN
                chunk = chunk.reindex(columns=cols_lower)
                total_rows += chunk[['worldx', 'worldy']].dropna().shape[0]
        except pd.errors.EmptyDataError:
            skipped_files.append((fpath, 'pandas EmptyDataError'))
        except Exception as e:
            skipped_files.append((fpath, f'Error during count: {e}'))

    print(f'counted total_rows={total_rows}')
    if total_rows == 0:
        print("No valid points found. Summary of skipped files:")
        for f, reason in skipped_files:
            print(f" - {f}: {reason}")
        raise SystemExit(1)

    # create memmap on disk (float64) in the system temp directory to avoid
    # issues with long/OneDrive paths on Windows
    if os.path.exists(MEMMAP_PATH):
        try:
            os.remove(MEMMAP_PATH)
        except Exception:
            pass
    try:
        mm = np.memmap(MEMMAP_PATH, dtype='float64', mode='w+', shape=(total_rows, 4))
    except OSError as e:
        print(f"Failed to create memmap at {MEMMAP_PATH}: {e}")
        raise
    print_mem('after memmap allocate')

    # fill memmap sequentially to avoid large intermediates
    idx = 0
    for fpath in csv_files:
        try:
            for chunk in pd.read_csv(fpath, usecols=['WorldX', 'WorldY'], chunksize=CHUNKSIZE):
                # normalize column names and reindex to canonical lower-case names
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk = chunk.reindex(columns=cols_lower)
                valid = chunk[['worldx', 'worldy']].dropna()
                if valid.shape[0] == 0:
                    continue
                # build a full 4-column array (worldx, worldy, lgvx, lgvy), fill missing with NaN
                a = valid.reindex(columns=cols_lower).to_numpy(dtype='float64')
                n = a.shape[0]
                if n:
                    mm[idx:idx+n, :] = a
                    idx += n
        except pd.errors.EmptyDataError:
            # already recorded in counting pass, but safe to skip
            continue
        except Exception as e:
            skipped_files.append((fpath, f'Error during fill: {e}'))

    # flush to disk and use memmap as the points array
    mm.flush()
    points = mm
    print_mem('after fill')
else:
    # legacy parallel read (keeps behavior if memmap disabled)
    if USE_PROCESS_POOL and len(csv_files) > 1:
        with ProcessPoolExecutor(max_workers=min(8, max(1, len(csv_files)))) as exc:
            futures = {exc.submit(read_coords_wrapper, (fpath, USE_PYARROW)): fpath for fpath in csv_files}
            for fut in as_completed(futures):
                coords, skip = fut.result()
                if skip:
                    skipped_files.append(skip)
                else:
                    points.append(coords)
    else:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(csv_files)))) as exc:
            futures = {exc.submit(read_coords, fpath): fpath for fpath in csv_files}
            for fut in as_completed(futures):
                coords, skip = fut.result()
                if skip:
                    skipped_files.append(skip)
                else:
                    points.append(coords)

io_time = time.time() - start_io
print(f"CSV read phase time: {io_time:.2f}s (files={len(csv_files)})")

# Optional: show memory usage if psutil is available
try:
    import psutil
    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / 1024**2
    print(f"Memory (RSS) after IO: {mem_mb:.1f} MB")
except Exception:
    pass

# Memory-friendly ingestion: if your dataset is huge, use a memmap to avoid building a large
# in-memory array. This performs a two-pass approach: count rows, create memmap, then fill it.
USE_MEMMAP = True
MEMMAP_PATH = os.path.join(csv_folder, 'coords.memmap')

if USE_MEMMAP and len(csv_files) > 0:
    # count total rows across csvs (only for files that weren't skipped)
    total_rows = 0
    valid_files = []
    for fpath in csv_files:
        try:
            if os.path.getsize(fpath) == 0:
                continue
            # fast row count: iterate with chunksize
            for chunk in pd.read_csv(fpath, usecols=['WorldX', 'WorldY'], chunksize=100000):
                total_rows += len(chunk.dropna())
            valid_files.append(fpath)
        except Exception:
            # skip problematic files
            continue

    if total_rows == 0:
        print("No valid points found after counting rows.")
        raise SystemExit(1)

    # create memmap and fill (4 columns: WorldX, WorldY, LgvX, LgvY)
    dtype = np.float64
    coords_shape = (total_rows, 4)
    coords_mm = np.memmap(MEMMAP_PATH, dtype=dtype, mode='w+', shape=coords_shape)
    idx = 0
    for fpath in valid_files:
        try:
            for chunk in pd.read_csv(fpath, usecols=lambda c: c in ['WorldX', 'WorldY', 'LgvX', 'LgvY'], chunksize=100000):
                    cols = ['WorldX', 'WorldY', 'LgvX', 'LgvY']
                    # normalize column whitespace
                    chunk.columns = [c.strip() for c in chunk.columns]
                    # ensure all desired columns exist (missing ones become NaN)
                    chunk = chunk.reindex(columns=cols)
                    # drop rows missing WorldX/WorldY but keep LGV columns aligned
                    valid = chunk.dropna(subset=['WorldX', 'WorldY'])
                    if valid.shape[0] == 0:
                        continue
                    arr = valid[cols].to_numpy(dtype=dtype)
                    if arr.size == 0:
                        continue
                    n = arr.shape[0]
                    coords_mm[idx:idx+n, :] = arr
                    idx += n
        except Exception:
            continue

    # flush memmap and use it as points (view)
    coords_mm.flush()

    # Convert memmap to ndarray view for inspection
    arr = np.asarray(coords_mm)

    # diagnostic: print memmap shape and first rows so we can see which columns populated
    try:
        print('memmap shape:', arr.shape)
        if arr.shape[0] > 0:
            print('memmap sample (first 5 rows):')
            # show up to 5 rows and all columns
            print(arr[:5, :])
    except Exception:
        pass

    try:
        mem_mb = proc.memory_info().rss / 1024**2
        print(f"Memory (RSS) after memmap fill: {mem_mb:.1f} MB")
    except Exception:
        pass

    # If Lgv columns (cols 2 and 3) are present (not all NaN), apply positional correction
    has_lgv = arr.shape[1] >= 4 and not np.all(np.isnan(arr[:, 2:4]))

    def correctPointPos(events, reflector_radius=0):
        """Offset point positions based on LGV heading and reflector radius
        events: iterable of (lgv_id, timestamp, wx, wy, lgvx, lgvy)
        Returns: Nx2 numpy array of corrected positions
        """
        corrected_points = []
        for event in events:
            # event may be (None,None,wx,wy,lgvx,lgvy) or (wx,wy,lgvx,lgvy)
            if len(event) >= 6:
                _, _, wx, wy, lgvx, lgvy = event
            else:
                wx, wy, lgvx, lgvy = event[0], event[1], event[2], event[3]

            reflector = np.array([wx, wy], dtype=float)
            lgv_pos = np.array([lgvx, lgvy], dtype=float)

            heading = reflector - lgv_pos  # Vector from LGV to reflector
            norm = np.linalg.norm(heading)
            if norm <= 0 or not np.isfinite(norm):
                unit_vector = np.array([0.0, 0.0])
            else:
                unit_vector = heading / norm

            corrected_pos = reflector + unit_vector * reflector_radius
            corrected_points.append(corrected_pos)

        return np.array(corrected_points)

    if has_lgv:
        # build events list from rows that have finite LGV positions
        valid_mask = np.isfinite(arr[:, 2]) & np.isfinite(arr[:, 3])
        events = []
        for row in arr[valid_mask]:
            wx, wy, lgvx, lgvy = row[0], row[1], row[2], row[3]
            events.append((None, None, wx, wy, lgvx, lgvy))

        print(f"Applying LGV correction to {len(events)} points")
        if len(events) == 0:
            print("No valid LGV positions found; falling back to raw WorldX/WorldY")
            points = arr[:, :2]
        else:
            points = correctPointPos(events, reflector_radius=0)
            # diagnostic: show first 5 originals vs corrected
            try:
                print('original first 5 WorldX,WorldY:')
                print(arr[:5, :2])
                print('corrected first 5 x,y:')
                print(points[:5, :])
            except Exception:
                pass
    else:
        points = arr[:, :2]

else:
    points = np.vstack(points)

# Run DBSCAN and measure time
start_cluster = time.time()

# Ensure `points` is a numpy array and check number of rows explicitly
if isinstance(points, np.ndarray):
    n_points = points.shape[0]
elif isinstance(points, (list, tuple)):
    n_points = sum(arr.shape[0] for arr in points) if points else 0
else:
    n_points = 0

if n_points == 0:
    print("No valid points found. Summary of skipped files:")
    for f, reason in skipped_files:
        print(f" - {f}: {reason}")
    raise SystemExit(1)

# If points is a list of arrays, stack them now (memmap path already produced ndarray)
if not isinstance(points, np.ndarray):
    points = np.vstack(points)

# Run DBSCAN (use the simple single-threaded instantiation — avoids potential slow-downs
# from parallel neighbor-search implementations on some platforms/versions)
dbscan = DBSCAN(eps=3.5, min_samples=20)

labels = dbscan.fit_predict(points)
cluster_time = time.time() - start_cluster
print(f"DBSCAN clustering time: {cluster_time:.2f}s (n_points={len(points)})")

# Prepare colors for clusters
def random_color():
    return [random.random() for _ in range(3)]

unique_labels = set(labels)
colors = {label: random_color() for label in unique_labels if label != -1}
colors[-1] = [0.2, 0.2, 0.2]  # dark gray for outliers

# Plotting: increase clustered point size a little, keep outliers ~30% size
# For very large point counts, rasterizing the scatter can speed rendering in matplotlib
cluster_marker_size = 14
outlier_marker_size = max(1, int(cluster_marker_size * 0.3))

start_plot = time.time()
num_points = points.shape[0]

plt.figure(figsize=(10, 8))
# Always plot points (no hexbin). Use slightly larger, rasterized markers and draw
# a semi-transparent yellow circle around each non-outlier cluster to highlight it.
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')
for label in unique_labels:
    mask = labels == label
    size = outlier_marker_size if label == -1 else cluster_marker_size
    # plot points on top
    plt.scatter(points[mask, 0], points[mask, 1], c=[colors[label]], s=size, alpha=0.7,
                edgecolors='none', rasterized=True, marker='.', zorder=2)

# Draw cluster circles behind the points
from matplotlib.patches import Circle
for label in unique_labels:
    if label == -1:
        continue
    mask = labels == label
    if np.count_nonzero(mask) == 0:
        continue
    cluster_pts = points[mask]
    centroid = cluster_pts.mean(axis=0)
    dists = np.linalg.norm(cluster_pts - centroid, axis=1)
    radius = float(np.percentile(dists, 90)) * 50.0
    circ = Circle((centroid[0], centroid[1]), radius=radius,
                  facecolor='red', alpha=0.25, edgecolor='darkred', linewidth=1, zorder=1)
    ax.add_patch(circ)

plt.xlabel('WorldX')
plt.ylabel('WorldY')
plt.title('DBSCAN Clusters of World Coordinates')
plt.tight_layout()
plot_time = time.time() - start_plot
print(f"Plotting time: {plot_time:.2f}s")
plt.show()

# Suggestions for further speedups with many points:
# - Downsample points before plotting (random sampling or density-aware sampling).
# - Use Datashader for rendering millions of points quickly (works with Bokeh/HoloViews).
# - Use interactive WebGL plotting (Plotly with scattergl) to leverage GPU rendering.
# - Use faster CSV readers (pyarrow) or binary pre-processed formats (Parquet) to reduce IO.