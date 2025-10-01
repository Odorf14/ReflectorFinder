import sys
import ezdxf
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


class DXFViewer(QGraphicsView):
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
        self.progress_dialog = None
        
        # Initialize empty scene with dark background
        self.setBackgroundBrush(QColor(10, 10, 10))

    def load_dxf(self, path):
        """Load and display DXF file"""
        print(f"[INFO] Loading DXF file: {path}")
        
        # Clear existing DXF content (keep dots)
        items_to_remove = []
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.scene.removeItem(item)
        
        try:
            doc = ezdxf.readfile(path)
        except Exception as e:
            print(f"[ERROR] Failed to read DXF: {e}")
            return
        
        print("[INFO] Parsing modelspace...")
        msp = doc.modelspace()
        pen = QPen(QColor(100, 100, 100))
        pen.setWidth(0)
        entity_count = 0

        # Batch add items to scene for better performance
        line_items = []
        for entity in msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                line = QGraphicsLineItem(start.x, start.y, end.x, end.y)
                line.setPen(pen)
                line.setZValue(0)
                line_items.append(line)
                entity_count += 1
                
                # Add in batches to reduce overhead
                if len(line_items) >= self.batch_size:
                    for item in line_items:
                        self.scene.addItem(item)
                    line_items.clear()
        
        # Add remaining items
        for item in line_items:
            self.scene.addItem(item)
        
        print(f"[INFO] Loaded {entity_count} LINE entities.")
        
        bbox = self.scene.itemsBoundingRect()
        print(f"[DEBUG] Scene bounds: {bbox}")
        
        self.resetTransform()
        self.fitInView(bbox, Qt.KeepAspectRatio)
        self.setTransform(self.transform().scale(1, -1))
        print("[INFO] DXF loading complete.")
        
    def load_csv_files_async(self, csv_paths):
        """Load multiple CSV files asynchronously with progress dialog"""
        if self.csv_loader_thread and self.csv_loader_thread.isRunning():
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Loading CSV files...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start background loading
        self.csv_loader_thread = CSVLoaderThread(csv_paths)
        self.csv_loader_thread.progress.connect(self.progress_dialog.setValue)
        self.csv_loader_thread.file_loaded.connect(self.on_csv_file_loaded)
        self.csv_loader_thread.finished_loading.connect(self.on_all_csv_loaded)
        self.csv_loader_thread.start()
        
    def on_csv_file_loaded(self, filename, points, events, correct_points = False):
        """Handle individual CSV file loaded"""
        print(f"[INFO] Processing {filename} with {len(points)} points...")
        
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
        dot_radius = 15  # Smaller radius for dense data
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
        self.progress_dialog.close()
        
        if len(self.all_points) == 0:
            print("[WARNING] No valid data loaded from any files.")
            return
        
        # Fit view to show all data
        bbox = self.scene.itemsBoundingRect()
        self.fitInView(bbox, Qt.KeepAspectRatio)
        
        print(f"[INFO] Loaded total of {len(self.all_points)} data points from all files.")

    def clear_dots(self):
        """Clear all dots from the scene"""
        print("[INFO] Clearing dots...")
        
        # Remove dot items in batches for better performance
        for i in range(0, len(self.dot_items), self.batch_size):
            batch = self.dot_items[i:i + self.batch_size]
            for item in batch:
                self.scene.removeItem(item)
        
        self.dot_items.clear()
        self.all_points = np.array([], dtype=np.float32)
        self.all_events.clear()
        print("[INFO] Cleared all dots and stored data.")
        
    def clear_reflectors(self):
        """Clear reflector visualizations from the scene"""
        print("[INFO] Clearing reflector visualizations...")
        
        for item in self.reflector_items:
            self.scene.removeItem(item)
        
        self.reflector_items.clear()
        print("[INFO] Cleared reflector visualizations.")

    def load_layoutReflectors(self, db3_path):
        """Load reflectors from db3"""
        conn = sqlite3.connect(db3_path)
        cursor = conn.cursor()

        cursor.execute("SELECT ID, X, Y FROM Reflectors")
        rows = cursor.fetchall()

        conn.close()
        self.visualize_layoutReflectors([(row[1], row[2]) for row in rows])
    
    def clear_layoutReflectors(self):
        """Clear layout reflector visualizations from the scene"""
        print("[INFO] Clearing layout reflector visualizations...")
        
        for item in self.layoutReflector_items:
            self.scene.removeItem(item)
        
        self.layoutReflector_items.clear()
        print("[INFO] Cleared layout reflector visualizations.")
        
    def visualize_reflectors(self, reflector_scores):
        """Create yellow circles for found reflectors with hover tooltips"""
        # Clear existing reflector visualizations
        self.clear_reflectors()
        
        if not reflector_scores:
            print("[INFO] No reflectors to visualize.")
            return
        
        circle_radius = 750  # Radius for reflector circles
        center_dot_radius = 32  # Small white center dot - actual reflector
        
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
        
        print(f"[INFO] Visualized {len(reflector_scores)} reflectors as yellow circles with white centers.")
        
    def visualize_layoutReflectors(self, layoutReflectors):
        """Create blue dots for layout reflectors"""
        dot_radius = 100
        dot_pen = QPen()
        dot_pen.setWidth(0)
        
        # Blue color for all dots
        blue_color = QColor(0, 0, 250, 255) 
        dot_pen.setColor(blue_color)
        
        # Create dots in batches
        layoutReflector_items_to_add = []
        
        for x, y in layoutReflectors:
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


    def wheelEvent(self, event):
        """Handle mouse wheel zoom with proper anchor point"""
        # Get the position of the mouse in scene coordinates
        old_pos = self.mapToScene(event.pos())

        # Zoom factor
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8

        # Apply zoom
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.scale(zoom_factor, zoom_factor)

        # After zoom, map the mouse position again and pan to keep it stable
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())


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
        self.splitter.addWidget(self.viewer)
        
        # Create console output area
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)  # Limit height
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
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

        # Load DXF action
        load_dxf_action = QAction("Load DXF", self)
        load_dxf_action.triggered.connect(self.load_dxf_file)
        file_menu.addAction(load_dxf_action)

        #Load layout reflectors action
        load_layoutReflectors_action = QAction("Load reflectors from layout", self)
        load_layoutReflectors_action.triggered.connect(self.load_layoutReflectors_file)
        file_menu.addAction(load_layoutReflectors_action)
        
        file_menu.addSeparator()

        # Load CSV action
        load_csv_action = QAction("Load Logs", self)
        load_csv_action.triggered.connect(self.load_csv_files)
        file_menu.addAction(load_csv_action)
        
        # Clear dots action
        clear_dots_action = QAction("Clear Points", self)
        clear_dots_action.triggered.connect(self.viewer.clear_dots)
        file_menu.addAction(clear_dots_action)
        
        # Clear reflectors action
        clear_reflectors_action = QAction("Clear Reflectors", self)
        clear_reflectors_action.triggered.connect(self.viewer.clear_reflectors)
        file_menu.addAction(clear_reflectors_action)

        #Clear layout reflectors action
        clear_layoutReflectors_action = QAction("Clear Layout Reflectors", self)
        clear_layoutReflectors_action.triggered.connect(self.viewer.clear_layoutReflectors)
        file_menu.addAction(clear_layoutReflectors_action)
        
        file_menu.addSeparator()
        
        # Clear console action
        clear_console_action = QAction("Clear Console", self)
        clear_console_action.triggered.connect(self.clear_console)
        file_menu.addAction(clear_console_action)
        
        # ANALYSIS MENU
        analysis_menu = menubar.addMenu("Analysis")
        
        # Placeholder for future reflector finding functionality
        find_reflectors_action = QAction("Find Reflectors", self)
        find_reflectors_action.triggered.connect(self.find_reflectors_placeholder)
        analysis_menu.addAction(find_reflectors_action)

    def clear_console(self):
        """Clear the console output area"""
        self.console.clear()
        self.console.setPlainText("Console Output:\n" + "="*50)

    def load_dxf_file(self):
        """Load DXF file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DXF File", "", "DXF Files (*.dxf)")
        if file_path:
            self.viewer.load_dxf(file_path)

    def load_csv_files(self):
        """Load multiple CSV files using optimized async loader"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV Files", "", "CSV Files (*.csv)")
        if file_paths:
            self.viewer.load_csv_files_async(file_paths)

    def load_layoutReflectors_file(self):
        """Load reflectors from db3"""

        file_path, _ = QFileDialog.getOpenFileName(self, "Open Layout.db3", "", "DB3 Files (*.db3)")
        if file_path:
            self.viewer.load_layoutReflectors(file_path)
            
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
            confidence_threshold=0.8,
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
            visual_msg = "Yellow circles have been added to the map. Hover over them for details."
            
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
        color: white;
    }
    QProgressBar {
        background-color: #3A3A3A;
        color: white;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #505050;
    }
    """
    app.setStyleSheet(dark_style)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())