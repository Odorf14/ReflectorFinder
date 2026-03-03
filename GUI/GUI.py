import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction,
    QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem,
    QWidget, QVBoxLayout, QProgressDialog, QTextEdit, QSplitter
)
from PyQt5.QtGui import QPen, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
from ReflectorFinder import (analyzeReflectors, correctPointPos)
import sqlite3
import xml.etree.ElementTree as ET
import os


class CSVLoaderThread(QThread):
    """Background thread for loading CSV files"""
    progress = pyqtSignal(int)
    file_loaded = pyqtSignal(str, list, list)  # filename, points, events
    finished_loading = pyqtSignal()
    
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        
    def run(self):
        total_files = len(self.file_paths)
        
        for i, csv_path in enumerate(self.file_paths):
            points, events = self.load_single_csv(csv_path)
            if points:  # Only emit if data was loaded
                self.file_loaded.emit(csv_path, points, events)
            
            progress_percent = int((i + 1) * 100 / total_files)
            self.progress.emit(progress_percent)
        
        self.finished_loading.emit()
    
    def load_single_csv(self, csv_path):
        """Load CSV data focusing on WorldX/WorldY coordinates"""
        points = []
        events = []

        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if we have the expected columns
                if reader.fieldnames:
                    expected_cols = ["Lgv", "Timestamp", "WorldX", "WorldY", "LgvX", "LgvY"]
                    missing_cols = [col for col in expected_cols if col not in reader.fieldnames]
                    if missing_cols:
                        print(f"[WARNING] Missing columns in {csv_path}: {missing_cols}")
                        print(f"[DEBUG] Available columns: {reader.fieldnames}")
                
                row_count = 0
                valid_rows = 0
                
                # Process rows efficiently
                for row in reader:
                    row_count += 1
                    try:
                        lgv_id = int(row['Lgv'])
                        timestamp = row['Timestamp']
                        worldx = float(row['WorldX'])
                        worldy = float(row['WorldY'])
                        lgvx = float(row['LgvX'])
                        lgvy = float(row['LgvY'])
                        
                        points.append((worldx, worldy))
                        events.append((lgv_id, timestamp, worldx, worldy, lgvx, lgvy))
                        valid_rows += 1
                        
                    except (ValueError, KeyError) as e:
                        # Log first few errors for debugging
                        if row_count <= 3:
                            print(f"[DEBUG] Row {row_count} error in {csv_path}: {e}")
                            print(f"[DEBUG] Row data: {dict(row)}")
                        continue
                
                if row_count > 0:
                    print(f"[INFO] {csv_path}: {valid_rows}/{row_count} valid rows processed")
                else:
                    print(f"[WARNING] {csv_path}: No rows found in file")
                        
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_path}: {e}")
            
        return points, events


class CSVLoaderThreadTC2(QThread):
    """Background thread for loading CSV files with TC2 filtering (layer-based)"""
    progress = pyqtSignal(int)
    file_loaded = pyqtSignal(str, list, list)  # filename, points, events
    finished_loading = pyqtSignal()
    
    def __init__(self, file_paths, freeshapes):
        super().__init__()
        self.file_paths = file_paths
        self.freeshapes = freeshapes  # List of (ID, X1, X2, Y1, Y2)
        
    def run(self):
        total_files = len(self.file_paths)
        
        for i, csv_path in enumerate(self.file_paths):
            points, events = self.load_single_csv_with_filter(csv_path)
            if points:  # Only emit if data was loaded
                self.file_loaded.emit(csv_path, points, events)
            
            progress_percent = int((i + 1) * 100 / total_files)
            self.progress.emit(progress_percent)
        
        self.finished_loading.emit()
    
    def point_in_freeshape(self, x, y, x1, x2, y1, y2):
        """Check if point (x, y) is inside the rectangle defined by (x1, x2, y1, y2)"""
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def find_containing_freeshape(self, x, y):
        """Find the FreeShape ID that contains point (x, y), returns None if not found"""
        for freeshape in self.freeshapes:
            shape_id, x1, x2, y1, y2 = freeshape
            if self.point_in_freeshape(x, y, x1, x2, y1, y2):
                return shape_id
        return None
    
    def load_single_csv_with_filter(self, csv_path):
        """Load CSV data with TC2 filtering based on FreeShapes"""
        points = []
        events = []

        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if we have the expected columns
                if reader.fieldnames:
                    expected_cols = ["Lgv", "Timestamp", "WorldX", "WorldY", "LgvX", "LgvY"]
                    missing_cols = [col for col in expected_cols if col not in reader.fieldnames]
                    if missing_cols:
                        print(f"[WARNING] Missing columns in {csv_path}: {missing_cols}")
                        print(f"[DEBUG] Available columns: {reader.fieldnames}")
                
                row_count = 0
                valid_rows = 0
                filtered_rows = 0
                
                # Process rows efficiently
                for row in reader:
                    row_count += 1
                    try:
                        lgv_id = int(row['Lgv'])
                        timestamp = row['Timestamp']
                        worldx = float(row['WorldX'])
                        worldy = float(row['WorldY'])
                        lgvx = float(row['LgvX'])
                        lgvy = float(row['LgvY'])
                        
                        # TC2 Filtering: Check if LGV position and detected reflector are in same layer
                        lgv_layer = self.find_containing_freeshape(lgvx, lgvy)
                        
                        if lgv_layer is not None:
                            # LGV is in a layer, check if detected reflector is in the same layer
                            reflector_layer = self.find_containing_freeshape(worldx, worldy)
                            
                            if reflector_layer == lgv_layer:
                                # Both in same layer, keep the point
                                points.append((worldx, worldy))
                                events.append((lgv_id, timestamp, worldx, worldy, lgvx, lgvy))
                                valid_rows += 1
                            else:
                                filtered_rows += 1
                        else:
                            # LGV not in any layer, filter out
                            filtered_rows += 1
                        
                    except (ValueError, KeyError) as e:
                        # Log first few errors for debugging
                        if row_count <= 3:
                            print(f"[DEBUG] Row {row_count} error in {csv_path}: {e}")
                            print(f"[DEBUG] Row data: {dict(row)}")
                        continue
                
                if row_count > 0:
                    print(f"[INFO] {csv_path}: {valid_rows}/{row_count} valid rows (filtered out: {filtered_rows})")
                else:
                    print(f"[WARNING] {csv_path}: No rows found in file")
                        
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_path}: {e}")
            
        return points, events


class DXFViewer(QGraphicsView):
    status_updated = pyqtSignal(str)  # Emitted to push messages to the console

    def __init__(self):
        super().__init__()
        self.all_events = []
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Store all points for future analysis
        self.all_points = np.array([], dtype=np.float32)
        self.dot_items = []
        self.reflector_items = []  # Store reflector visualization items
        self.layoutReflector_items = []  # Store layout reflector visualization items
        
        # Batch processing for graphics items
        self.batch_size = 1000
        
        # CSV loading thread
        self.csv_loader_thread = None
        self.csv_loader_thread_tc2 = None
        self.progress_dialog = None
        
        # Database layout data
        self.layout_reflectors = []  # Store reflectors from db3
        self.layout_freeshapes = []  # Store FreeShapes from db3
        self.background_items = []   # Store background geometry items from db3
        self.db3_loaded = False  # Flag to track if db3 has been loaded

        # Drawing sizes – overridden by MainWindow after construction
        self.log_radius = 15
        self.db3_radius = 100
        self.reflector_radius = 32
        self.highlight_radius = 1750
        
        # Initialize empty scene with dark background
        self.setBackgroundBrush(QColor(50, 50, 50))

    def load_csv_files_async(self, csv_paths):
        """Load multiple CSV files asynchronously with progress dialog (TC3)"""
        if self.csv_loader_thread and self.csv_loader_thread.isRunning():
            return
        
        msg = f"[INFO] Loading {len(csv_paths)} TC3 log file(s)..."
        print(msg)
        self.status_updated.emit(msg)

        # Show progress dialog
        self.progress_dialog = QProgressDialog("Loading CSV files (TC3)...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start background loading
        self.csv_loader_thread = CSVLoaderThread(csv_paths)
        self.csv_loader_thread.progress.connect(self.progress_dialog.setValue)
        self.csv_loader_thread.file_loaded.connect(self.on_csv_file_loaded)
        self.csv_loader_thread.finished_loading.connect(self.on_all_csv_loaded)
        self.csv_loader_thread.start()
    
    def load_csv_files_async_tc2(self, csv_paths):
        """Load multiple CSV files with TC2 filtering (layer-based)"""
        if self.csv_loader_thread_tc2 and self.csv_loader_thread_tc2.isRunning():
            return
        
        msg = f"[INFO] Loading {len(csv_paths)} TC2 log file(s) with layer filtering..."
        print(msg)
        self.status_updated.emit(msg)

        # Show progress dialog
        self.progress_dialog = QProgressDialog("Loading CSV files (TC2 with filtering)...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start background loading with filtering
        self.csv_loader_thread_tc2 = CSVLoaderThreadTC2(csv_paths, self.layout_freeshapes)
        self.csv_loader_thread_tc2.progress.connect(self.progress_dialog.setValue)
        self.csv_loader_thread_tc2.file_loaded.connect(self.on_csv_file_loaded)
        self.csv_loader_thread_tc2.finished_loading.connect(self.on_all_csv_loaded)
        self.csv_loader_thread_tc2.start()
        
    def on_csv_file_loaded(self, filename, points, events, correct_points = False):
        """Handle individual CSV file loaded"""
        short_name = os.path.basename(filename)
        msg = f"[INFO] Loaded {short_name}: {len(points)} points."
        print(msg)
        self.status_updated.emit(msg)
        
        # Store points for future analysis
        if len(self.all_points) == 0:
            self.all_points = np.array(points, dtype=np.float32)
        else:
            self.all_points = np.concatenate([self.all_points, np.array(points, dtype=np.float32)])
        
        self.all_events.extend(events)
        
        #Adjusting offset assuming every point is a real reflector.
        #Only for visualization, the adjustment is done again during analysis.
        if correct_points: points = correctPointPos(events, reflector_radius=32)

        # Create red dots for all points
        self.create_red_dots(points)
        
    def create_red_dots(self, points):
        """Create small red dots for all points"""
        dot_radius = self.log_radius
        dot_pen = QPen()
        dot_pen.setWidth(0)
        
        # Red color for all dots
        red_color = QColor(255, 0, 0, 180)  # Semi-transparent red
        dot_pen.setColor(red_color)
        
        # Create dots in batches
        dot_items_to_add = []
        
        for x, y in points:
            ellipse = QGraphicsEllipseItem(x - dot_radius / 2, y - dot_radius / 2, dot_radius, dot_radius)
            ellipse.setPen(dot_pen)
            ellipse.setBrush(red_color)
            ellipse.setZValue(1)
            
            self.dot_items.append(ellipse)
            dot_items_to_add.append(ellipse)
            
            # Add to scene in batches
            if len(dot_items_to_add) >= self.batch_size:
                for item in dot_items_to_add:
                    self.scene.addItem(item)
                dot_items_to_add.clear()

        # Add remaining items
        for item in dot_items_to_add:
            self.scene.addItem(item)
        
    def on_all_csv_loaded(self):
        """Handle completion of all CSV loading"""
        if self.progress_dialog:
            self.progress_dialog.close()
        
        if len(self.all_points) == 0:
            msg = "[WARNING] No valid data loaded from any files."
            print(msg)
            self.status_updated.emit(msg)
            return
        
        # Fit view to show all data
        bbox = self.scene.itemsBoundingRect()
        self.fitInView(bbox, Qt.KeepAspectRatio)
        
        msg = f"[INFO] Done. Total logged points loaded: {len(self.all_points):,}."
        print(msg)
        self.status_updated.emit(msg)

    def clear_dots(self):
        """Clear all dots from the scene"""
        count = len(self.dot_items)
        
        # Remove dot items in batches for better performance
        for i in range(0, count, self.batch_size):
            batch = self.dot_items[i:i + self.batch_size]
            for item in batch:
                self.scene.removeItem(item)
        
        self.dot_items.clear()
        self.all_points = np.array([], dtype=np.float32)
        self.all_events.clear()

        msg = f"[INFO] Cleared {count:,} logged points."
        print(msg)
        self.status_updated.emit(msg)
        
    def clear_reflectors(self):
        """Clear reflector visualizations from the scene"""
        count = len(self.reflector_items)
        for item in self.reflector_items:
            self.scene.removeItem(item)
        self.reflector_items.clear()

        msg = f"[INFO] Cleared {count} found reflector markers."
        print(msg)
        self.status_updated.emit(msg)

    def load_layout(self, db3_path):
        """Load reflectors, FreeShapes and background geometry from db3"""
        # Auto-clear any previously loaded layout before loading a new one
        if self.db3_loaded:
            self.clear_layout()

        label = os.path.basename(db3_path)

        # Progress dialog – 4 steps: Reflectors, FreeShapes, Background, Fit
        progress = QProgressDialog(f"Loading layout: {label}", None, 0, 4, self.window())
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            conn = sqlite3.connect(db3_path)
            cursor = conn.cursor()

            progress.setLabelText("Loading reflectors...")
            progress.setValue(1)
            QApplication.processEvents()
            cursor.execute("SELECT ID, X, Y FROM Reflectors")
            reflector_rows = cursor.fetchall()
            self.layout_reflectors = reflector_rows

            progress.setLabelText("Loading layers (FreeShapes)...")
            progress.setValue(2)
            QApplication.processEvents()
            cursor.execute("SELECT ID, X1, X2, Y1, Y2 FROM FreeShapes")
            freeshape_rows = cursor.fetchall()
            self.layout_freeshapes = freeshape_rows

            progress.setLabelText("Loading background geometry...")
            progress.setValue(3)
            QApplication.processEvents()
            self.load_background_from_db(conn)

            conn.close()
        except Exception as e:
            progress.close()
            msg = f"[ERROR] Failed to load layout: {e}"
            print(msg)
            self.status_updated.emit(msg)
            return

        self.db3_loaded = True
        self.visualize_layout_reflectors([(row[1], row[2]) for row in reflector_rows])

        progress.setLabelText("Fitting view...")
        progress.setValue(4)
        QApplication.processEvents()

        bbox = self.scene.itemsBoundingRect()
        if not bbox.isEmpty():
            margin = max(bbox.width(), bbox.height()) * 5
            self.scene.setSceneRect(bbox.adjusted(-margin, -margin, margin, margin))
            self.resetTransform()
            self.fitInView(bbox, Qt.KeepAspectRatio)
            self.setTransform(self.transform().scale(1, -1))
            self.centerOn(bbox.center())

        progress.close()

        msg = (
            f"[INFO] Layout loaded: {len(reflector_rows)} reflectors, "
            f"{len(freeshape_rows)} layers, "
            f"{len(self.background_items)} background entities."
        )
        print(msg)
        self.status_updated.emit(msg)
    
    def clear_layout_reflectors(self):
        """Clear only the blue db3 reflector dots, keeping background geometry"""
        count = len(self.layoutReflector_items)
        for item in self.layoutReflector_items:
            self.scene.removeItem(item)
        self.layoutReflector_items.clear()

        msg = f"[INFO] Cleared {count} db3 reflector markers."
        print(msg)
        self.status_updated.emit(msg)

    def clear_layout(self):
        """Clear layout reflector visualizations, background geometry and db3 data from the scene"""
        n_reflectors = len(self.layoutReflector_items)
        n_bg = len(self.background_items)

        for item in self.layoutReflector_items:
            self.scene.removeItem(item)
        self.layoutReflector_items.clear()

        for item in self.background_items:
            self.scene.removeItem(item)
        self.background_items.clear()

        self.layout_reflectors = []
        self.layout_freeshapes = []
        self.db3_loaded = False

        msg = f"[INFO] Layout cleared ({n_reflectors} reflector markers, {n_bg} background entities removed)."
        print(msg)
        self.status_updated.emit(msg)

    def load_background_from_db(self, conn):
        """Load and draw background geometry from BackGround and DxfPolylinePoints tables"""
        pen = QPen(QColor(170, 170, 170))
        pen.setWidth(0)

        counts = {1: 0, 2: 0, 3: 0, 4: 0}

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT Type, Data1, Data2, Data3, Data4, Data5 FROM BackGround")
            rows = cursor.fetchall()
        except Exception as e:
            print(f"[ERROR] Failed to query BackGround table: {e}")
            return

        for row in rows:
            entity_type, d1, d2, d3, d4, d5 = row
            try:
                if entity_type == 1:
                    # Line: (X1, Y1) -> (X2, Y2)
                    item = self.scene.addLine(float(d1), float(d2), float(d3), float(d4), pen)
                    item.setZValue(0)
                    self.background_items.append(item)
                    counts[1] += 1
                elif entity_type == 2:
                    # Circle: center (d1, d2), radius d3
                    cx, cy, r = float(d1), float(d2), float(d3)
                    item = self.scene.addEllipse(cx - r, cy - r, 2 * r, 2 * r, pen)
                    item.setZValue(0)
                    self.background_items.append(item)
                    counts[2] += 1
                elif entity_type == 3:
                    # Polyline: first PointId = d2, num points = d3
                    self._draw_bg_polyline(conn, int(d2), int(d3), pen)
                    counts[3] += 1
                elif entity_type == 4:
                    # Insert: skip
                    counts[4] += 1
            except Exception as e:
                print(f"[ERROR] BackGround entity type {entity_type}: {e}")

        print(
            f"[INFO] Background geometry: Lines={counts[1]}, Circles={counts[2]}, "
            f"Polylines={counts[3]}, Inserts(skipped)={counts[4]}"
        )

    def _draw_bg_polyline(self, conn, first_point_id, num_points, pen):
        """Query DxfPolylinePoints and draw consecutive line segments"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT X, Y FROM DxfPolylinePoints "
                "WHERE PointId >= ? AND PointId < ? ORDER BY PointId",
                (first_point_id, first_point_id + num_points),
            )
            points = cursor.fetchall()
        except Exception as e:
            print(f"[ERROR] DxfPolylinePoints query failed: {e}")
            return

        for i in range(len(points) - 1):
            x1, y1 = float(points[i][0]),     float(points[i][1])
            x2, y2 = float(points[i + 1][0]), float(points[i + 1][1])
            item = self.scene.addLine(x1, y1, x2, y2, pen)
            item.setZValue(0)
            self.background_items.append(item)
        
    def visualize_reflectors(self, reflector_scores):
        """Create yellow circles for found reflectors with hover tooltips"""
        # Clear existing reflector visualizations
        self.clear_reflectors()
        
        if not reflector_scores:
            print("[INFO] No reflectors to visualize.")
            return
        
        circle_radius = self.highlight_radius
        center_dot_radius = self.reflector_radius
        
        yellow_color = QColor(255, 255, 0, 100)  # Semi-transparent yellow
        yellow_pen = QPen(QColor(255, 255, 0, 200), 3)  # Yellow border
        
        white_color = QColor(255, 255, 255, 255)  # Opaque white
        white_pen = QPen(QColor(255, 255, 255, 255), 2)  # White border
        
        for cluster_id, centroid, confidence in reflector_scores:
            x, y = centroid[0], centroid[1]
            # Create main yellow circle
            circle = QGraphicsEllipseItem(
                x - circle_radius, y - circle_radius,
                circle_radius * 2, circle_radius * 2
            )
            circle.setPen(yellow_pen)
            circle.setBrush(yellow_color)
            circle.setZValue(10)  # Above everything else
            
            # Create small white center dot
            center_dot = QGraphicsEllipseItem(
                x - center_dot_radius, y - center_dot_radius,
                center_dot_radius * 2, center_dot_radius * 2
            )
            center_dot.setPen(white_pen)
            center_dot.setBrush(white_color)
            center_dot.setZValue(11)  # Above the yellow circle
            
            # Set tooltip for both items
            tooltip_text = (f"Reflector {cluster_id}\n"
                          f"Confidence: {confidence:.3f}\n"
                          f"X: {x:.1f}\n"
                          f"Y: {y:.1f}")
            circle.setToolTip(tooltip_text)
            center_dot.setToolTip(tooltip_text)
            
            # Add to scene and store references
            self.scene.addItem(circle)
            self.scene.addItem(center_dot)
            self.reflector_items.append(circle)
            self.reflector_items.append(center_dot)
        
        msg = f"[INFO] Found reflectors visualized: {len(reflector_scores)} candidate(s)."
        print(msg)
        self.status_updated.emit(msg)
        
    def visualize_layout_reflectors(self, layout_reflectors):
        """Create blue dots for layout reflectors"""
        dot_radius = self.db3_radius
        dot_pen = QPen()
        dot_pen.setWidth(0)
        
        # Blue color for all dots
        blue_color = QColor(80, 80, 250, 255) 
        dot_pen.setColor(blue_color)
        
        # Create dots in batches
        layoutReflector_items_to_add = []
        
        for x, y in layout_reflectors:
            ellipse = QGraphicsEllipseItem(x - dot_radius / 2, y - dot_radius / 2, dot_radius, dot_radius)
            ellipse.setPen(dot_pen)
            ellipse.setBrush(blue_color)
            ellipse.setZValue(2)
            
            self.layoutReflector_items.append(ellipse)
            layoutReflector_items_to_add.append(ellipse)
            
            # Add to scene in batches
            if len(layoutReflector_items_to_add) >= self.batch_size:
                for item in layoutReflector_items_to_add:
                    self.scene.addItem(item)
                layoutReflector_items_to_add.clear()

        # Add remaining items
        for item in layoutReflector_items_to_add:
            self.scene.addItem(item)

        msg = f"[INFO] Layout reflectors drawn: {len(layout_reflectors)} marker(s)."
        print(msg)
        self.status_updated.emit(msg)


    def wheelEvent(self, event):
        """Handle mouse wheel zoom anchored to the mouse pointer (free zoom)"""
        # Snapshot mouse position in scene space before scaling
        old_pos = self.mapToScene(event.pos())

        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8

        # Scale with no view anchor so we control panning manually
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.scale(zoom_factor, zoom_factor)

        # Translate so the scene point under the mouse stays fixed
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        # Expand the scene rect to prevent Qt from snapping/re-centering
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        current = self.sceneRect()
        if not current.contains(visible):
            self.setSceneRect(current.united(visible).adjusted(
                -current.width(), -current.height(),
                current.width(), current.height()
            ))

def get_config_path():
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    xml_path = os.path.join(exe_dir, 'ReflectorFinder')
    os.makedirs(xml_path, exist_ok=True)
    print(f"[WARNING] Loading config from: {xml_path}")
    return os.path.join(xml_path, 'ReflectorFinder.xml')

def generate_configfile():
    """
    Generate a default configuration XML file with standard settings.
    Creates ReflectorFinder.xml with default GUI and analysisparameters.
    """
    root = ET.Element("Settings")

    dbs = ET.SubElement(root, "DBSCAN")
    ET.SubElement(dbs, "EPS").text = '3.5'
    ET.SubElement(dbs, "MinSamples").text = '30'


    confidence = ET.SubElement(root, "ConfidenceScores")
    ET.SubElement(confidence, "ConfidenceCheck").text = 'True'
    ET.SubElement(confidence, "MinConfidence").text = '0.8'


    freq_score = ET.SubElement(confidence, "FrequencyScore")
    ET.SubElement(freq_score, "Weight").text = '0.15'
    ET.SubElement(freq_score, "MaxFreqThreshold").text = '50'

    lgv_div_score = ET.SubElement(confidence, "LGVDiversityScore")
    ET.SubElement(lgv_div_score, "Weight").text = '0.35'
    ET.SubElement(lgv_div_score, "MaxLGVThreshold").text = '10'

    timestamp_score = ET.SubElement(confidence, "TimestampScore")
    ET.SubElement(timestamp_score, "Weight").text = '0.25'
    ET.SubElement(timestamp_score, "MaxTimeVarianceSec").text = '7200'

    spatial_score = ET.SubElement(confidence, "SpatialDistributionScore")
    ET.SubElement(spatial_score, "Weight").text = '0.25'
    ET.SubElement(spatial_score, "MaxSpatialStdDev").text = '10.0'


    gui = ET.SubElement(root, "GUI_Settings")
    log_reflector = ET.SubElement(gui, "Logged_Points")
    ET.SubElement(log_reflector, "PointRadius").text = '15' #dot_radius

    db3_reflector = ET.SubElement(gui, "DB3_Reflector")
    ET.SubElement(db3_reflector, "PointRadius").text = '100'#dot_radius

    found_reflector = ET.SubElement(gui, "Found_Reflector")
    ET.SubElement(found_reflector, "PointRadius").text = '32' #center_dot_radius
    ET.SubElement(found_reflector, "HighlightRadius").text = '1750' #circle_radius


    ET.indent(root, space="  ", level=0)

    config_path = get_config_path()
    tree = ET.ElementTree(root)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated config file at: {config_path}")

def load_configuration():
    """
    Load configuration from XML file. Generates default config if file doesn't exist.
    Returns:
        tuple: (config_created, plc_ams_id, plc_ip, port, tc2, remoterun_enable, one_file, log_duration, read_interval, 
                log_folder, max_log_size, symbols_tc3, symbols_tc2)
        symbols is a dict mapping symbol names to (symbol_path, bypass_flag) tuples
    """
    #Normalizar weights
    #Corregir docstring

    config_path = get_config_path()
    config_created = False

    if not os.path.exists(config_path):
        print("[WARNING] Configuration file not found. Generating default config.")
        generate_configfile()
        config_created = True

    tree = ET.parse(config_path)
    root = tree.getroot()

    eps = float(root.find("DBSCAN/EPS").text)
    min_samples = int(root.find("DBSCAN/MinSamples").text)
    confidence_check = root.find("ConfidenceScores/ConfidenceCheck").text.lower() == 'true'
    min_confidence = float(root.find("ConfidenceScores/MinConfidence").text)
    freq_weight = float(root.find("ConfidenceScores/FrequencyScore/Weight").text)
    max_freq_threshold = float(root.find("ConfidenceScores/FrequencyScore/MaxFreqThreshold").text)
    lgv_div_weight = float(root.find("ConfidenceScores/LGVDiversityScore/Weight").text)
    max_lgv_threshold = float(root.find("ConfidenceScores/LGVDiversityScore/MaxLGVThreshold").text)
    timestamp_weight = float(root.find("ConfidenceScores/TimestampScore/Weight").text)
    max_time_variance = float(root.find("ConfidenceScores/TimestampScore/MaxTimeVarianceSec").text)
    spatial_weight = float(root.find("ConfidenceScores/SpatialDistributionScore/Weight").text)
    max_spatial_stddev = float(root.find("ConfidenceScores/SpatialDistributionScore/MaxSpatialStdDev").text)
    log_radius = int(root.find("GUI_Settings/Logged_Points/PointRadius").text)
    db3_radius = int(root.find("GUI_Settings/DB3_Reflector/PointRadius").text)
    reflector_radius = int(root.find("GUI_Settings/Found_Reflector/PointRadius").text)
    highlight_radius = int(root.find("GUI_Settings/Found_Reflector/HighlightRadius").text)

    return (config_created, eps, min_samples, confidence_check, min_confidence,
            freq_weight, max_freq_threshold, lgv_div_weight, max_lgv_threshold,
            timestamp_weight, max_time_variance, spatial_weight, max_spatial_stddev,
            log_radius, db3_radius, reflector_radius, highlight_radius)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reflector Finder")

        # Create splitter to divide main view and console
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Create layout for main widget
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter (vertical split)
        self.splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.splitter)
        
        # Create and add viewer to top part of splitter
        self.viewer = DXFViewer()
        self.viewer.status_updated.connect(self.log_to_console)
        # Apply initial config sizes to the viewer
        self.viewer.log_radius      = log_radius
        self.viewer.db3_radius      = db3_radius
        self.viewer.reflector_radius = reflector_radius
        self.viewer.highlight_radius = highlight_radius
        self.splitter.addWidget(self.viewer)
        
        # Create console output area
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)  # Limit height
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #323232;
                color: #C8C8C8;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #555;
            }
        """)
        self.console.setPlainText("Console Output:\n" + "="*50)
        self.splitter.addWidget(self.console)
        
        # Set splitter proportions (80% viewer, 20% console)
        self.splitter.setSizes([800, 200])
        self.splitter.setCollapsible(1, True)  # Console can be collapsed

        # Setup menu
        self.init_menu()
        
        self.destroyed.connect(QApplication.quit)

    def init_menu(self):
        """Initialize menu bar with basic functionality"""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("File")

        #Load layout db3 action
        load_layout_action = QAction("Load layout", self)
        load_layout_action.triggered.connect(self.load_layout_file)
        file_menu.addAction(load_layout_action)

        # Load CSV TC3 action
        load_csv_tc3_action = QAction("Load Logs TC3", self)
        load_csv_tc3_action.triggered.connect(self.load_csv_files_tc3)
        file_menu.addAction(load_csv_tc3_action)
        
        # Load CSV TC2 action
        load_csv_tc2_action = QAction("Load Logs TC2", self)
        load_csv_tc2_action.triggered.connect(self.load_csv_files_tc2)
        file_menu.addAction(load_csv_tc2_action)

        file_menu.addSeparator()
        
        # Clear dots action
        clear_dots_action = QAction("Clear Points", self)
        clear_dots_action.triggered.connect(self.viewer.clear_dots)
        file_menu.addAction(clear_dots_action)
        
        # Clear reflectors action
        clear_reflectors_action = QAction("Clear Reflectors", self)
        clear_reflectors_action.triggered.connect(self.viewer.clear_reflectors)
        file_menu.addAction(clear_reflectors_action)

        #Clear db3 action
        clear_layout_action = QAction("Clear layout", self)
        clear_layout_action.triggered.connect(self.viewer.clear_layout)
        file_menu.addAction(clear_layout_action)

        # Clear db3 reflectors only action
        clear_layout_reflectors_action = QAction("Clear db3 reflectors", self)
        clear_layout_reflectors_action.triggered.connect(self.viewer.clear_layout_reflectors)
        file_menu.addAction(clear_layout_reflectors_action)
        
        file_menu.addSeparator()
        
        # Clear console action
        clear_console_action = QAction("Clear Console", self)
        clear_console_action.triggered.connect(self.clear_console)
        file_menu.addAction(clear_console_action)
        
        # ANALYSIS MENU
        analysis_menu = menubar.addMenu("Analysis")
        
        # Find TC2 Reflectors
        find_tc2_reflectors_action = QAction("Find TC2 Reflectors", self)
        find_tc2_reflectors_action.triggered.connect(self.find_tc2_reflectors)
        analysis_menu.addAction(find_tc2_reflectors_action)
        
        # Find TC3 Reflectors
        find_tc3_reflectors_action = QAction("Find TC3 Reflectors", self)
        find_tc3_reflectors_action.triggered.connect(self.find_reflectors_placeholder)
        analysis_menu.addAction(find_tc3_reflectors_action)
        
        # Reload configuration action
        reload_config_action = QAction("Reload configuration", self)
        reload_config_action.triggered.connect(self.reload_configuration)
        analysis_menu.addAction(reload_config_action)

    def clear_console(self):
        """Clear the console output area"""
        self.console.clear()
        self.console.setPlainText("Console Output:\n" + "="*50)
    
    def reload_configuration(self):
        """Reload configuration from XML file"""
        start_msg = "[INFO] Reloading configuration..."
        print(start_msg)
        self.log_to_console(start_msg)
        
        try:
            # Call load_configuration to reload settings
            config_created, eps, min_samples, confidence_check, min_confidence, freq_weight, max_freq_threshold, \
            lgv_div_weight, max_lgv_threshold, timestamp_weight, max_time_variance, spatial_weight, max_spatial_stddev, \
            log_radius, db3_radius, reflector_radius, highlight_radius = load_configuration()
            
            # Update global variables (if needed in the future)
            globals()['eps'] = eps
            globals()['min_samples'] = min_samples
            globals()['confidence_check'] = confidence_check
            globals()['min_confidence'] = min_confidence
            globals()['freq_weight'] = freq_weight
            globals()['max_freq_threshold'] = max_freq_threshold
            globals()['lgv_div_weight'] = lgv_div_weight
            globals()['max_lgv_threshold'] = max_lgv_threshold
            globals()['timestamp_weight'] = timestamp_weight
            globals()['max_time_variance'] = max_time_variance
            globals()['spatial_weight'] = spatial_weight
            globals()['max_spatial_stddev'] = max_spatial_stddev
            globals()['log_radius'] = log_radius
            globals()['db3_radius'] = db3_radius
            globals()['reflector_radius'] = reflector_radius
            globals()['highlight_radius'] = highlight_radius

            # Push new sizes into the viewer so next draw uses them
            self.viewer.log_radius       = log_radius
            self.viewer.db3_radius       = db3_radius
            self.viewer.reflector_radius = reflector_radius
            self.viewer.highlight_radius = highlight_radius
            
            success_msg = "[INFO] Configuration reloaded successfully."
            print(success_msg)
            self.log_to_console(success_msg)
            
        except Exception as e:
            error_msg = f"[ERROR] Failed to reload configuration: {e}"
            print(error_msg)
            self.log_to_console(error_msg)

    def load_csv_files_tc3(self):
        """Load multiple CSV files using optimized async loader (TC3 - no filtering)"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV Files", "", "CSV Files (*.csv)")
        if file_paths:
            self.viewer.load_csv_files_async(file_paths)
    
    def load_csv_files_tc2(self):
        """Load multiple CSV files with TC2 layer-based filtering"""
        # Check if db3 has been loaded
        if not self.viewer.db3_loaded:
            error_msg = "[ERROR] Load layout first before loading TC2 logs."
            print(error_msg)
            self.log_to_console(error_msg)
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV Files (TC2)", "", "CSV Files (*.csv)")
        if file_paths:
            self.viewer.load_csv_files_async_tc2(file_paths)

    def load_layout_file(self):
        """Load layout reflectors, FreeShapes and background from db3"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Layout.db3", "", "DB3 Files (*.db3)")
        if file_path:
            self.viewer.load_layout(file_path)
    
    def find_tc2_reflectors(self):
        """Find TC2 Reflectors - placeholder for future implementation"""
        if not self.viewer.db3_loaded:
            warning_msg = "[WARNING] Load layout first before finding TC2 reflectors."
            print(warning_msg)
            self.log_to_console(warning_msg)
            return
        
        # Placeholder for future TC2 reflector finding logic
        info_msg = "[INFO] TC2 Reflector finding functionality will be implemented here."
        print(info_msg)
        self.log_to_console(info_msg)
            
    def find_reflectors_placeholder(self):
        """Run reflector finding analysis and display results"""
        if len(self.viewer.all_points) == 0:
            error_msg = "[ERROR] No data loaded. Please load CSV files first."
            print(error_msg)
            self.log_to_console(error_msg)
            return
        
        start_msg = "[INFO] Starting reflector analysis..."
        print(start_msg)
        self.log_to_console(start_msg)
        self.log_to_console("=" * 50)
        
        # Run the analysis
        reflector_scores = analyzeReflectors(
            self.viewer.all_points, 
            self.viewer.all_events,
            confidence_threshold=min_confidence,
            offset_correction=False
        )
        
        # Display console output
        self.log_to_console("\nREFLECTOR ANALYSIS RESULTS")
        self.log_to_console("=" * 50)
        
        if reflector_scores:
            self.log_to_console("Found reflectors:")
            for i, (cluster_id, centroid, confidence) in enumerate(reflector_scores, 1):
                x, y = centroid[0], centroid[1]
                result_line = f"Reflector {i}: Confidence: {confidence:.3f}, X: {x:.1f}, Y: {y:.1f}"
                print(result_line)
                self.log_to_console(result_line)
            
            # Visualize the reflectors
            self.viewer.visualize_reflectors(reflector_scores)
            
            summary_msg = f"\nTotal reflectors found: {len(reflector_scores)}"
            visual_msg = "Yellow circles highlighting found reflectors. Hover over them for details."
            
            print(summary_msg)
            print(visual_msg)
            self.log_to_console(summary_msg)
            self.log_to_console(visual_msg)
        else:
            no_results_msg = "No reflectors found with sufficient confidence."
            print(no_results_msg)
            self.log_to_console(no_results_msg)
            
        self.log_to_console("=" * 50)
        
    def log_to_console(self, message):
        """Add message to the embedded console"""
        self.console.append(message)
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def apply_dark_theme(app):
    """Apply dark theme to the application"""
    dark_style = """
    QMainWindow {
        background-color: #2E2E2E;
    }
    QMenuBar {
        background-color: #3A3A3A;
        color: white;
    }
    QMenuBar::item {
        background-color: #3A3A3A;
        color: white;
    }
    QMenuBar::item:selected {
        background-color: #505050;
    }
    QMenu {
        background-color: #3A3A3A;
        color: white;
    }
    QMenu::item:selected {
        background-color: #505050;
    }
    QScrollBar:vertical {
        background: #3A3A3A;
        width: 12px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:vertical {
        background: #707070;
        min-height: 20px;
        border-radius: 6px;
    }
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        background: none;
        height: 0px;
    }
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {
        background: none;
    }
    QScrollBar:horizontal {
        background: #3A3A3A;
        height: 12px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:horizontal {
        background: #707070;
        min-width: 20px;
        border-radius: 6px;
    }
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        background: none;
        width: 0px;
    }
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {
        background: none;
    }
    QProgressDialog {
        background-color: #2E2E2E;
        color: #E0E0E0;
    }
    QProgressDialog QLabel {
        color: #E0E0E0;
        font-size: 10pt;
    }
    QProgressBar {
        background-color: #3A3A3A;
        color: #E0E0E0;
        text-align: center;
        font-size: 10pt;
        font-weight: bold;
    }
    QProgressBar::chunk {
        background-color: #505050;
    }
    """
    app.setStyleSheet(dark_style)


if __name__ == "__main__":
    config_created, eps, min_samples, confidence_check, min_confidence, freq_weight, max_freq_threshold, \
    lgv_div_weight, max_lgv_threshold, timestamp_weight, max_time_variance, spatial_weight, max_spatial_stddev, \
    log_radius, db3_radius, reflector_radius, highlight_radius = load_configuration()

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())