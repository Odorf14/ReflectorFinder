import sys
import csv
import os
import time
import ctypes
import math
from datetime import datetime
import pyads
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction,
    QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem,
    QGraphicsPolygonItem, QGraphicsSimpleTextItem,
    QWidget, QVBoxLayout, QProgressDialog, QTextEdit, QSplitter,
    QDialog, QFormLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QGridLayout, QScrollArea, QGroupBox, QDialogButtonBox,
    QSplashScreen
)
from PyQt5.QtGui import QPen, QPainter, QColor, QPolygonF, QFont, QTransform, QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
import numpy as np
from ReflectorFinder import (analyzeReflectors, correctPointPos)
import sqlite3
import xml.etree.ElementTree as ET


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


class BackgroundLoaderThread(QThread):
    """
    Reads ALL geometry from BackGround + DxfPolylinePoints in a background thread.
    Key optimisation: loads the entire DxfPolylinePoints table into a dict once,
    eliminating the N individual sub-queries that caused slow loads on dense layouts.
    Emits raw coordinate data; QGraphicsItems are created on the main thread.
    """
    progress = pyqtSignal(int, int)          # (processed, total)
    finished = pyqtSignal(list, list, dict)  # (line_segs, ellipses, counts)
    log      = pyqtSignal(str)

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def run(self):
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. How many entities to process (drives progress bar)
            total = cursor.execute("SELECT COUNT(*) FROM BackGround").fetchone()[0]
            self.progress.emit(0, total)

            # 2. Preload ALL polyline points in one shot  (dict: PointId → (x, y))
            poly_points = {}
            for pid, x, y in cursor.execute(
                "SELECT PointId, X, Y FROM DxfPolylinePoints ORDER BY PointId"
            ):
                poly_points[int(pid)] = (float(x), float(y))

            # 3. Iterate BackGround and build flat lists of primitives
            line_segs = []  # (x1, y1, x2, y2)
            ellipses  = []  # (x, y, w, h) in bounding-box form
            counts    = {1: 0, 2: 0, 3: 0, 4: 0}

            for i, row in enumerate(cursor.execute(
                "SELECT Type, Data1, Data2, Data3, Data4, Data5 FROM BackGround"
            )):
                entity_type, d1, d2, d3, d4, _d5 = row
                try:
                    if entity_type == 1:
                        line_segs.append((float(d1), float(d2), float(d3), float(d4)))
                        counts[1] += 1
                    elif entity_type == 2:
                        cx, cy, r = float(d1), float(d2), float(d3)
                        ellipses.append((cx - r, cy - r, 2.0 * r, 2.0 * r))
                        counts[2] += 1
                    elif entity_type == 3:
                        first_id = int(d2)
                        num_pts  = int(d3)
                        pts = [poly_points[pid]
                               for pid in range(first_id, first_id + num_pts)
                               if pid in poly_points]
                        for j in range(len(pts) - 1):
                            line_segs.append((pts[j][0], pts[j][1],
                                              pts[j + 1][0], pts[j + 1][1]))
                        if len(pts) >= 2:
                            counts[3] += 1
                    elif entity_type == 4:
                        counts[4] += 1
                except (ValueError, TypeError) as e:
                    self.log.emit(f"[WARNING] BackGround row {i} (type {entity_type}): {e}")

                if i % 500 == 0:
                    self.progress.emit(i, total)

            self.progress.emit(total, total)
            conn.close()
            self.finished.emit(line_segs, ellipses, counts)

        except Exception as e:
            self.log.emit(f"[ERROR] BackgroundLoaderThread: {e}")
            self.finished.emit([], [], {})


######################################################
# TC3 Structs (shared between EKFReaderThread and any future TC3 consumers)
######################################################

class _ReflectorObs(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("rho",       ctypes.c_float),
        ("phi",       ctypes.c_float),
        ("x",         ctypes.c_float),
        ("y",         ctypes.c_float),
        ("radius",    ctypes.c_float),
        ("quality",   ctypes.c_float),
    ]

class _ReflectorLandmark(ctypes.Structure):
    _fields_ = [
        ("id",     ctypes.c_int32),
        ("x",      ctypes.c_float),
        ("y",      ctypes.c_float),
        ("radius", ctypes.c_float),
        ("hMin",   ctypes.c_float),
        ("hMax",   ctypes.c_float),
    ]

class _ReflectorInfo(ctypes.Structure):
    _fields_ = [
        ("obs",          _ReflectorObs),
        ("landmark",     _ReflectorLandmark),
        ("wrtAgvX",      ctypes.c_float),
        ("wrtAgvY",      ctypes.c_float),
        ("worldX_preUpd",ctypes.c_float),
        ("worldY_preUpd",ctypes.c_float),
        ("worldX",       ctypes.c_float),
        ("worldY",       ctypes.c_float),
        ("updateLag",    ctypes.c_float),
        ("associated",   ctypes.c_uint8),
        ("pad",          ctypes.c_byte * 3),
    ]

_NUM_TC3_REFLECTORS = 50
_ReflectorArray = _ReflectorInfo * _NUM_TC3_REFLECTORS
_REFLECTOR_RAW_SIZE = ctypes.sizeof(_ReflectorArray)


######################################################
# EKF Settings Dialog
######################################################

class EKFSettingsDialog(QDialog):
    def __init__(self, ekf_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EKF Viewer — Connection Settings")
        self.setMinimumWidth(540)
        self._original_config = {
            "ams_net_id": ekf_config.get("ams_net_id", ""),
            "port": ekf_config.get("port", 851),
            "read_interval_ms": ekf_config.get("read_interval_ms", 1000),
            "symbols": dict(ekf_config.get("symbols", {})),
        }

        outer = QVBoxLayout(self)

        # --- Connection group ---
        conn_group = QGroupBox("Connection")
        conn_layout = QFormLayout(conn_group)

        self.ams_edit = QLineEdit(ekf_config.get("ams_net_id", "192.168.11.2.1.1"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(ekf_config.get("port", 851))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 60000)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setValue(ekf_config.get("read_interval_ms", 1000))

        conn_layout.addRow("AMS Net ID:", self.ams_edit)
        conn_layout.addRow("Port:", self.port_spin)
        conn_layout.addRow("Read Interval:", self.interval_spin)
        outer.addWidget(conn_group)

        # --- Symbols group (scrollable) ---
        syms_group = QGroupBox("TC3 Symbols")
        syms_layout = QFormLayout(syms_group)

        symbols = ekf_config.get("symbols", {})
        self._symbol_edits = {}
        self._bypass_checks = {}

        _bypass_sym_names = {"AvoidReflectorCheck", "Quality", "Aut_Run", "Man_Run", "IsNotMoving"}
        _all_sym_order = [
            "AvoidReflectorCheck", "Quality", "Aut_Run", "Man_Run", "IsNotMoving",
            "LgvPosX", "LgvPosY", "LgvPosH", "NumLGV", "ForwMotion", "Reflectors",
        ]
        for sym_name in _all_sym_order:
            sym_path, bypass = symbols.get(sym_name, ("", False))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            edit = QLineEdit(sym_path)
            row_layout.addWidget(edit)
            self._symbol_edits[sym_name] = edit

            if sym_name in _bypass_sym_names:
                chk = QCheckBox("Bypass")
                chk.setChecked(bypass)
                row_layout.addWidget(chk)
                self._bypass_checks[sym_name] = chk

            syms_layout.addRow(f"{sym_name}:", row_widget)

        scroll = QScrollArea()
        scroll.setWidget(syms_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(320)
        outer.addWidget(scroll)

        # --- Buttons ---
        btn_box = QDialogButtonBox()
        btn_box.addButton("Start", QDialogButtonBox.AcceptRole)
        btn_box.addButton(QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        outer.addWidget(btn_box)

    def get_config(self):
        symbols = {}
        for sym_name, edit in self._symbol_edits.items():
            bypass_chk = self._bypass_checks.get(sym_name)
            bypass = bypass_chk.isChecked() if bypass_chk else False
            symbols[sym_name] = (edit.text().strip(), bypass)
        return {
            "ams_net_id": self.ams_edit.text().strip(),
            "port": self.port_spin.value(),
            "read_interval_ms": self.interval_spin.value(),
            "symbols": symbols,
        }

    def config_changed(self):
        return self.get_config() != self._original_config


######################################################
# Analysis Settings Dialog
######################################################

class AnalysisSettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        self.setMinimumWidth(560)

        outer = QVBoxLayout(self)

        # --- DBSCAN group ---
        dbscan_group = QGroupBox("DBSCAN")
        dbscan_layout = QFormLayout(dbscan_group)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.1, 1000.0)
        self.eps_spin.setDecimals(2)
        self.eps_spin.setSingleStep(0.5)
        self.eps_spin.setValue(cfg.get("eps", 3.5))

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 100000)
        self.min_samples_spin.setValue(cfg.get("min_samples", 30))

        dbscan_layout.addRow("EPS:", self.eps_spin)
        dbscan_layout.addRow("Min Samples:", self.min_samples_spin)
        outer.addWidget(dbscan_group)

        # --- Confidence Filter group ---
        conf_group = QGroupBox("Confidence Filter")
        conf_layout = QFormLayout(conf_group)

        self.conf_check = QCheckBox()
        self.conf_check.setChecked(cfg.get("confidence_check", True))

        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setDecimals(2)
        self.min_conf_spin.setSingleStep(0.05)
        self.min_conf_spin.setValue(cfg.get("min_confidence", 0.8))

        conf_layout.addRow("Enable Confidence Check:", self.conf_check)
        conf_layout.addRow("Min Confidence:", self.min_conf_spin)
        outer.addWidget(conf_group)

        # --- Score Weights & Thresholds group (grid: weight | threshold per row) ---
        weights_group = QGroupBox("Score Weights && Thresholds")
        grid = QGridLayout(weights_group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _dspin(val, lo=0.0, hi=1.0, decimals=2, step=0.05):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(val)
            return s

        self.freq_weight_spin    = _dspin(cfg.get("freq_weight", 0.15))
        self.max_freq_spin       = _dspin(cfg.get("max_freq_threshold", 50.0),  lo=1.0, hi=10000.0, decimals=1, step=5.0)
        self.lgv_div_weight_spin = _dspin(cfg.get("lgv_div_weight", 0.35))
        self.max_lgv_spin        = _dspin(cfg.get("max_lgv_threshold", 10.0),   lo=1.0, hi=1000.0,  decimals=1, step=1.0)
        self.ts_weight_spin      = _dspin(cfg.get("timestamp_weight", 0.25))
        self.max_time_spin       = _dspin(cfg.get("max_time_variance", 7200.0), lo=1.0, hi=86400.0, decimals=0, step=600.0)
        self.spatial_weight_spin = _dspin(cfg.get("spatial_weight", 0.25))
        self.max_spatial_spin    = _dspin(cfg.get("max_spatial_stddev", 10.0),  lo=0.1, hi=10000.0, decimals=1, step=1.0)

        # Header row
        hdr_weight = QLabel("<b>Weight</b>")
        hdr_thresh = QLabel("<b>Max Threshold</b>")
        hdr_weight.setAlignment(Qt.AlignCenter)
        hdr_thresh.setAlignment(Qt.AlignCenter)
        grid.addWidget(hdr_weight, 0, 1)
        grid.addWidget(hdr_thresh, 0, 3)

        rows = [
            ("Frequency",           self.freq_weight_spin,    "Max Frequency",         self.max_freq_spin),
            ("LGV Diversity",       self.lgv_div_weight_spin, "Max LGV Count",         self.max_lgv_spin),
            ("Timestamp",           self.ts_weight_spin,      "Max Time Variance (s)", self.max_time_spin),
            ("Spatial Distribution",self.spatial_weight_spin, "Max Spatial Std Dev",   self.max_spatial_spin),
        ]
        for r, (wlabel, wspin, tlabel, tspin) in enumerate(rows, start=1):
            grid.addWidget(QLabel(wlabel + ":"),  r, 0)
            grid.addWidget(wspin,                 r, 1)
            grid.addWidget(QLabel(tlabel + ":"),  r, 2)
            grid.addWidget(tspin,                 r, 3)

        outer.addWidget(weights_group)

        # --- Warning label ---
        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: #ff6b6b; font-style: italic;")
        self._warning_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._warning_label)

        # --- Buttons ---
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_button = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        outer.addWidget(btn_box)

        # Connect weight spinboxes to validation
        for wspin in (self.freq_weight_spin, self.lgv_div_weight_spin,
                      self.ts_weight_spin, self.spatial_weight_spin):
            wspin.valueChanged.connect(self._validate_weights)

        self._validate_weights()

    def _validate_weights(self):
        total = round(
            self.freq_weight_spin.value() +
            self.lgv_div_weight_spin.value() +
            self.ts_weight_spin.value() +
            self.spatial_weight_spin.value(),
            6
        )
        ok = abs(total - 1.0) < 1e-4
        self._ok_button.setEnabled(ok)
        if ok:
            self._warning_label.setText("")
        else:
            self._warning_label.setText(
                f"Weights must sum to 1.00 (current sum: {total:.4f})"
            )

    def get_config(self):
        return {
            "eps":               self.eps_spin.value(),
            "min_samples":       self.min_samples_spin.value(),
            "confidence_check":  self.conf_check.isChecked(),
            "min_confidence":    self.min_conf_spin.value(),
            "freq_weight":       self.freq_weight_spin.value(),
            "max_freq_threshold":self.max_freq_spin.value(),
            "lgv_div_weight":    self.lgv_div_weight_spin.value(),
            "max_lgv_threshold": self.max_lgv_spin.value(),
            "timestamp_weight":  self.ts_weight_spin.value(),
            "max_time_variance": self.max_time_spin.value(),
            "spatial_weight":    self.spatial_weight_spin.value(),
            "max_spatial_stddev":self.max_spatial_spin.value(),
        }


######################################################
# Visualization Settings Dialog
######################################################

class VisualizationSettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualization Settings")
        self.setMinimumWidth(360)

        outer = QVBoxLayout(self)

        def _ispin(val, lo=1, hi=100000):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            return s

        # --- Logged Points ---
        log_group = QGroupBox("Logged Points")
        log_layout = QFormLayout(log_group)
        self.log_radius_spin = _ispin(cfg.get("log_radius", 15))
        log_layout.addRow("Point Radius:", self.log_radius_spin)
        outer.addWidget(log_group)

        # --- DB3 Reflector ---
        db3_group = QGroupBox("DB3 Reflector")
        db3_layout = QFormLayout(db3_group)
        self.db3_radius_spin = _ispin(cfg.get("db3_radius", 100))
        db3_layout.addRow("Point Radius:", self.db3_radius_spin)
        outer.addWidget(db3_group)

        # --- Found Reflector ---
        found_group = QGroupBox("Found Reflector")
        found_layout = QFormLayout(found_group)
        self.reflector_radius_spin = _ispin(cfg.get("reflector_radius", 32))
        self.highlight_radius_spin = _ispin(cfg.get("highlight_radius", 1750))
        found_layout.addRow("Center Dot Radius:", self.reflector_radius_spin)
        found_layout.addRow("Highlight Circle Radius:", self.highlight_radius_spin)
        outer.addWidget(found_group)

        # --- Buttons ---
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        outer.addWidget(btn_box)

    def get_config(self):
        return {
            "log_radius":       self.log_radius_spin.value(),
            "db3_radius":       self.db3_radius_spin.value(),
            "reflector_radius": self.reflector_radius_spin.value(),
            "highlight_radius": self.highlight_radius_spin.value(),
        }


######################################################
# EKF Reader Thread
######################################################

class EKFReaderThread(QThread):
    cycle_data = pyqtSignal(float, float, float, list, list, bool, bool, float)
    # lgv_x_mm, lgv_y_mm, lgv_h_cdeg, associated_list, new_unassoc_list, conditions_met, forw_motion, quality
    lgv_number = pyqtSignal(int)
    connection_status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, ekf_config):
        super().__init__()
        self._config = ekf_config
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        cfg = self._config
        ams_net_id = cfg["ams_net_id"]
        plc_ip = '.'.join(ams_net_id.split('.')[:4])
        port = cfg["port"]
        interval_ms = cfg["read_interval_ms"]
        symbols = cfg["symbols"]

        avoid_bypass      = symbols["AvoidReflectorCheck"][1]
        quality_bypass    = symbols["Quality"][1]
        aut_run_bypass    = symbols["Aut_Run"][1]
        man_run_bypass    = symbols["Man_Run"][1]
        isnotmoving_bypass = symbols["IsNotMoving"][1]
        reflectors_sym    = symbols["Reflectors"][0]

        plc = pyads.Connection(ams_net_id, port, plc_ip)
        try:
            plc.open()
            self.connection_status.emit(f"[INFO] EKF Viewer: Connected to {ams_net_id}:{port}")
        except pyads.ADSError as e:
            self.connection_status.emit(f"[ERROR] EKF Viewer: Could not connect: {e}")
            self.finished.emit()
            return

        def _fetch_handles():
            h = {
                "lgv_x":     plc.get_symbol(symbols["LgvPosX"][0]),
                "lgv_y":     plc.get_symbol(symbols["LgvPosY"][0]),
                "lgv_h":     plc.get_symbol(symbols["LgvPosH"][0]),
                "avoid":     plc.get_symbol(symbols["AvoidReflectorCheck"][0]) if not avoid_bypass else None,
                "quality":   plc.get_symbol(symbols["Quality"][0])             if not quality_bypass else None,
                "aut_run":   plc.get_symbol(symbols["Aut_Run"][0])             if not aut_run_bypass else None,
                "man_run":   plc.get_symbol(symbols["Man_Run"][0])             if not man_run_bypass else None,
                "isnotmoving": plc.get_symbol(symbols["IsNotMoving"][0])       if not isnotmoving_bypass else None,
                "forw":        plc.get_symbol(symbols["ForwMotion"][0])        if "ForwMotion" in symbols else None,
                "reflectors":  plc.get_symbol(reflectors_sym),
            }
            return h

        try:
            handles = _fetch_handles()
        except pyads.ADSError as e:
            self.connection_status.emit(f"[ERROR] EKF Viewer: Failed to get symbol handles: {e}")
            try:
                plc.close()
            except Exception:
                pass
            self.finished.emit()
            return

        # Read LGV number once after connection (graceful if symbol absent or old config)
        try:
            if "NumLGV" in symbols:
                lgv_num_val = int(plc.get_symbol(symbols["NumLGV"][0]).read())
                self.lgv_number.emit(lgv_num_val)
        except Exception:
            pass

        previous_unassoc = []
        next_time = time.perf_counter()

        try:
            while not self._stop_flag:
                try:
                    lgv_x = handles["lgv_x"].read()
                    lgv_y = handles["lgv_y"].read()
                    lgv_h = handles["lgv_h"].read()
                    _refl = handles["reflectors"]
                    raw_data = plc.read(_refl.index_group, _refl.index_offset, ctypes.c_ubyte * _REFLECTOR_RAW_SIZE)
                    forw_motion = handles["forw"].read() if handles.get("forw") else True

                    avoid      = handles["avoid"].read()      if handles["avoid"]      else False
                    quality    = handles["quality"].read()    if handles["quality"]    else 0.0
                    aut_run    = handles["aut_run"].read()    if handles["aut_run"]    else False
                    man_run    = handles["man_run"].read()    if handles["man_run"]    else False
                    isnotmoving = handles["isnotmoving"].read() if handles["isnotmoving"] else False

                    avoidref_ok  = avoid_bypass      or not avoid
                    quality_ok   = quality_bypass    or quality > 0.8
                    run_ok       = (aut_run_bypass and man_run_bypass) or aut_run or man_run
                    notmoving_ok = isnotmoving_bypass or not isnotmoving
                    conditions_met = avoidref_ok and quality_ok and run_ok and notmoving_ok

                    reflectors = _ReflectorArray.from_buffer_copy(bytes(raw_data))

                    associated = [
                        (r.worldX, r.worldY, r.landmark.id)
                        for r in reflectors
                        if r.associated and r.worldX != 0.0
                    ]
                    unassoc_all = [
                        (r.worldX, r.worldY)
                        for r in reflectors
                        if not r.associated and r.worldX != 0.0
                    ]

                    if conditions_met:
                        new_unassoc = [pt for pt in unassoc_all if pt not in previous_unassoc]
                    else:
                        new_unassoc = []

                    previous_unassoc = unassoc_all

                    self.cycle_data.emit(lgv_x, lgv_y, lgv_h, associated, new_unassoc, conditions_met, forw_motion, quality)

                except pyads.ADSError as e:
                    self.connection_status.emit(f"[WARNING] EKF Viewer: Read error: {e}. Reconnecting...")
                    try:
                        plc.close()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    if self._stop_flag:
                        break
                    try:
                        plc.open()
                        handles = _fetch_handles()
                        self.connection_status.emit("[INFO] EKF Viewer: Reconnected.")
                    except pyads.ADSError as conn_err:
                        self.connection_status.emit(f"[ERROR] EKF Viewer: Reconnect failed: {conn_err}")
                    next_time = time.perf_counter()
                    continue

                except Exception as e:
                    self.connection_status.emit(f"[WARNING] EKF Viewer: Unexpected error: {e}")
                    time.sleep(1.0)
                    next_time = time.perf_counter()
                    continue

                next_time += interval_ms / 1000.0
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()

        finally:
            try:
                plc.close()
            except Exception:
                pass
            self.connection_status.emit("[INFO] EKF Viewer: Disconnected.")
            self.finished.emit()


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
        self.progress_dialog = None

        # Background geometry loader thread
        self._bg_loader = None
        
        # Database layout data
        self.layout_reflectors = []  # Store reflectors from db3
        self.background_items = []   # Store background geometry items from db3
        self.db3_loaded = False  # Flag to track if db3 has been loaded

        # Drawing sizes – overridden by MainWindow after construction
        self.log_radius = 15
        self.db3_radius = 100
        self.reflector_radius = 32
        self.highlight_radius = 1750
        
        # EKF Viewer state
        self._ekf_follow = False
        self.ekf_lgv_arrow = None
        self.ekf_lgv_quality_text = None
        self.ekf_associated_items = []
        self.ekf_unassociated_items = []
        self.ekf_unassociated_data = []   # list of (timestamp, x_mm, y_mm, lgv_x_mm, lgv_y_mm)
        self._ekf_zoom_on_first = False    # zoom in to LGV on first arrow update

        # Analysis state
        self.associated_reflector_ids = set()   # IDs loaded from associated-reflectors logs
        self.unused_reflector_items = []        # graphics items for never-seen reflectors

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
        """Load reflectors synchronously (fast), then kick off
        BackgroundLoaderThread for the heavy geometry work."""
        if self.db3_loaded:
            self.clear_layout()

        label = os.path.basename(db3_path)

        # --- Fast sync part: Reflectors ---
        try:
            conn = sqlite3.connect(db3_path)
            cursor = conn.cursor()
            cursor.execute("SELECT ID, X, Y FROM Reflectors")
            reflector_rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            msg = f"[ERROR] Failed to load layout metadata: {e}"
            print(msg)
            self.status_updated.emit(msg)
            return

        self.layout_reflectors = reflector_rows
        self.visualize_layout_reflectors(reflector_rows)

        msg = (f"[INFO] Metadata loaded: {len(reflector_rows)} reflectors. Loading background geometry...")
        print(msg)
        self.status_updated.emit(msg)

        # --- Async part: background geometry (can be slow for large layouts) ---
        self.progress_dialog = QProgressDialog(
            f"Reading background geometry: {label}", None, 0, 0, self.window()
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.show()
        QApplication.processEvents()

        self._bg_loader = BackgroundLoaderThread(db3_path)
        self._bg_loader.progress.connect(self._on_bg_read_progress)
        self._bg_loader.log.connect(lambda m: (print(m), self.status_updated.emit(m)))
        self._bg_loader.finished.connect(
            lambda segs, elps, cnts: self._on_background_loaded(
                segs, elps, cnts, reflector_rows
            )
        )
        self._bg_loader.start()

    def _on_bg_read_progress(self, current, total):
        """Update progress dialog while the background thread reads the database."""
        if not self.progress_dialog:
            return
        if total > 0:
            if self.progress_dialog.maximum() == 0:
                self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(
                f"Reading background geometry... {current:,} / {total:,} entities"
            )

    def _on_background_loaded(self, line_segs, ellipses, counts, reflector_rows):
        """Create QGraphicsItems on the main thread from raw data received from the loader thread."""
        pen = QPen(QColor(170, 170, 170))
        pen.setWidth(0)

        total_items = len(line_segs) + len(ellipses)

        if self.progress_dialog:
            self.progress_dialog.setMaximum(total_items if total_items > 0 else 1)
            self.progress_dialog.setValue(0)
            self.progress_dialog.setLabelText(f"Drawing {total_items:,} entities...")
            QApplication.processEvents()

        # Draw lines (includes polyline segments – already flattened by the thread)
        for i, (x1, y1, x2, y2) in enumerate(line_segs):
            item = self.scene.addLine(x1, y1, x2, y2, pen)
            item.setZValue(0)
            self.background_items.append(item)
            if i % 2000 == 0 and self.progress_dialog:
                self.progress_dialog.setValue(i)
                QApplication.processEvents()

        # Draw circles
        for j, (x, y, w, h) in enumerate(ellipses):
            item = self.scene.addEllipse(x, y, w, h, pen)
            item.setZValue(0)
            self.background_items.append(item)

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.db3_loaded = True

        count_msg = (
            f"[INFO] Background geometry: Lines={counts.get(1, 0)}, "
            f"Circles={counts.get(2, 0)}, Polylines={counts.get(3, 0)}, "
            f"Inserts(skipped)={counts.get(4, 0)}"
        )
        print(count_msg)
        self.status_updated.emit(count_msg)

        bbox = self.scene.itemsBoundingRect()
        if not bbox.isEmpty():
            margin = max(bbox.width(), bbox.height()) * 5
            self.scene.setSceneRect(bbox.adjusted(-margin, -margin, margin, margin))
            self.resetTransform()
            self.fitInView(bbox, Qt.KeepAspectRatio)
            self.ensure_y_flip()
            self.centerOn(bbox.center())

        msg = (
            f"[INFO] Layout loaded: {len(reflector_rows)} reflectors, "
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

    def clear_unused_reflectors(self):
        """Remove never-seen reflector markers from the scene."""
        for item in self.unused_reflector_items:
            self.scene.removeItem(item)
        self.unused_reflector_items.clear()

    def visualize_unused_reflectors(self, unused_reflectors):
        """Draw yellow markers with ID labels for reflectors never seen in associated logs.

        unused_reflectors: list of (id, x, y) tuples.
        """
        self.clear_unused_reflectors()

        if not unused_reflectors:
            return

        circle_radius = self.highlight_radius
        center_radius = self.reflector_radius

        yellow_brush = QColor(255, 200, 0, 80)
        yellow_pen   = QPen(QColor(255, 200, 0, 220), 3)
        white_brush  = QColor(255, 255, 255, 255)
        white_pen    = QPen(QColor(255, 255, 255, 255), 2)

        flip_tf = QTransform.fromScale(1, -1)
        label_font = QFont("Arial")
        label_font.setPointSizeF(max(1.0, center_radius * 2.5))

        for rid, x, y in unused_reflectors:
            tooltip = f"ID: {rid}\nX: {x:.1f}\nY: {y:.1f}\n(Never seen)"

            circle = QGraphicsEllipseItem(
                x - circle_radius, y - circle_radius,
                circle_radius * 2, circle_radius * 2
            )
            circle.setPen(yellow_pen)
            circle.setBrush(yellow_brush)
            circle.setZValue(10)
            circle.setToolTip(tooltip)

            center = QGraphicsEllipseItem(
                x - center_radius, y - center_radius,
                center_radius * 2, center_radius * 2
            )
            center.setPen(white_pen)
            center.setBrush(white_brush)
            center.setZValue(11)
            center.setToolTip(tooltip)

            label = QGraphicsSimpleTextItem(str(rid))
            label.setFont(label_font)
            label.setBrush(QColor(255, 255, 100))
            label.setTransform(flip_tf)
            label.setPos(x + center_radius * 1.5, y + center_radius * 0.5)
            label.setZValue(12)
            label.setToolTip(tooltip)

            self.scene.addItem(circle)
            self.scene.addItem(center)
            self.scene.addItem(label)
            self.unused_reflector_items.extend([circle, center, label])

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

        self.clear_unused_reflectors()

        self.layout_reflectors = []
        self.db3_loaded = False

        msg = f"[INFO] Layout cleared ({n_reflectors} reflector markers, {n_bg} background entities removed)."
        print(msg)
        self.status_updated.emit(msg)

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
        
        for i, (cluster_id, centroid, confidence) in enumerate(reflector_scores, 1):
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
            tooltip_text = (f"Reflector {i}\n"
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
        """Create blue dots for layout reflectors. layout_reflectors: list of (id, x, y) tuples."""
        dot_radius = self.db3_radius
        dot_pen = QPen()
        dot_pen.setWidth(0)
        
        # Blue color for all dots
        blue_color = QColor(80, 80, 250, 255) 
        dot_pen.setColor(blue_color)
        
        # Create dots in batches
        layoutReflector_items_to_add = []
        
        for rid, x, y in layout_reflectors:
            ellipse = QGraphicsEllipseItem(x - dot_radius / 2, y - dot_radius / 2, dot_radius, dot_radius)
            ellipse.setPen(dot_pen)
            ellipse.setBrush(blue_color)
            ellipse.setZValue(2)
            ellipse.setToolTip(f"ID: {rid}\nX: {x:.1f}\nY: {y:.1f}")
            
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

    def apply_visualization_resize(self, old_cfg, new_cfg):
        """Resize existing scene items to reflect updated visualization settings.
        Only touches items whose corresponding radius actually changed."""

        def _resize(item, new_diameter):
            """Keep the item's centre fixed; update bounding rect to new_diameter x new_diameter."""
            c = item.rect().center()
            h = new_diameter / 2.0
            item.setRect(c.x() - h, c.y() - h, new_diameter, new_diameter)

        log_changed  = old_cfg['log_radius']      != new_cfg['log_radius']
        db3_changed  = old_cfg['db3_radius']       != new_cfg['db3_radius']
        refl_changed = old_cfg['reflector_radius'] != new_cfg['reflector_radius']
        hl_changed   = old_cfg['highlight_radius'] != new_cfg['highlight_radius']

        if log_changed:
            d = new_cfg['log_radius']
            for item in self.dot_items:
                _resize(item, d)

        if db3_changed:
            d = new_cfg['db3_radius']
            for item in self.layoutReflector_items:
                _resize(item, d)

        if hl_changed or refl_changed:
            # reflector_items: pairs [highlight_circle, center_dot, ...]
            new_hl   = new_cfg['highlight_radius']
            new_refl = new_cfg['reflector_radius']
            for i in range(0, len(self.reflector_items) - 1, 2):
                if hl_changed:
                    _resize(self.reflector_items[i],     new_hl * 2)
                if refl_changed:
                    _resize(self.reflector_items[i + 1], new_refl * 2)

        if refl_changed:
            # EKF unassociated red dots: radius = reflector_radius * 4
            d = new_cfg['reflector_radius'] * 8
            for item in self.ekf_unassociated_items:
                _resize(item, d)

        if hl_changed or refl_changed:
            # unused_reflector_items: triples [outer_circle, center_dot, label, ...]
            new_hl   = new_cfg['highlight_radius']
            new_refl = new_cfg['reflector_radius']
            new_font = QFont("Arial")
            new_font.setPointSizeF(max(1.0, new_refl * 2.5))
            for i in range(0, len(self.unused_reflector_items), 3):
                outer  = self.unused_reflector_items[i]
                center = self.unused_reflector_items[i + 1]
                label  = self.unused_reflector_items[i + 2]
                c = center.rect().center()
                if hl_changed:
                    _resize(outer, new_hl * 2)
                if refl_changed:
                    _resize(center, new_refl * 2)
                    label.setFont(new_font)
                    label.setPos(c.x() + new_refl * 1.5, c.y() + new_refl * 0.5)

    # ------------------------------------------------------------------
    # EKF Viewer — view helpers
    # ------------------------------------------------------------------

    def ensure_y_flip(self):
        """Apply Y-axis flip to the view exactly once (idempotent)."""
        if self.transform().m22() > 0:
            self.setTransform(self.transform().scale(1, -1))

    def set_ekf_follow(self, enabled):
        """Enable/disable auto-centering on the LGV arrow each cycle."""
        self._ekf_follow = enabled

    def update_lgv_arrow(self, x_mm, y_mm, h_cdeg, forw_motion=True, quality=0.0):
        """Create (first call) or update the LGV position arrow.

        x_mm, y_mm:  world position in millimetres (scene units).
        h_cdeg:      heading in centidegrees; 0 = north (+Y), increases CW.
        forw_motion: True = forward (cyan/blue), False = reverse (pink).
        """
        self.ensure_y_flip()
        if self.ekf_lgv_arrow is None:
            # Triangle pointing in the +scene-Y direction (north after Y-flip).
            # Tip forward, two rear corners + small notch at rear centre.
            arrow_poly = QPolygonF([
                QPointF(   0.0,  900.0),   # tip (forward)
                QPointF(-330.0, -390.0),   # rear-left
                QPointF(   0.0,  -90.0),   # rear notch
                QPointF( 330.0, -390.0),   # rear-right
            ])
            self.ekf_lgv_arrow = QGraphicsPolygonItem(arrow_poly)
            self.ekf_lgv_arrow.setZValue(20)
            self.scene.addItem(self.ekf_lgv_arrow)

        # Update color every cycle based on forward/reverse direction
        if forw_motion:
            self.ekf_lgv_arrow.setBrush(QColor(0, 220, 255, 210))    # cyan-blue
            self.ekf_lgv_arrow.setPen(QPen(QColor(0, 170, 210), 60))
        else:
            self.ekf_lgv_arrow.setBrush(QColor(255, 110, 200, 210))  # pink
            self.ekf_lgv_arrow.setPen(QPen(QColor(220, 60, 170), 60))

        self.ekf_lgv_arrow.setPos(x_mm, y_mm)
        # h_cdeg is CW-from-north in centidegrees.  -90° aligns the polygon
        # (which points in +Y = north) with the actual forward direction.
        self.ekf_lgv_arrow.setRotation(h_cdeg * 0.01 - 90.0)

        # Quality label — rendered as a separate upright text item inside the arrow
        quality_pct = f"{quality * 100:.0f}%"
        if self.ekf_lgv_quality_text is None:
            self.ekf_lgv_quality_text = QGraphicsSimpleTextItem()
            font = QFont("Arial")
            font.setPointSizeF(180)
            font.setBold(True)
            self.ekf_lgv_quality_text.setFont(font)
            self.ekf_lgv_quality_text.setBrush(QColor(255, 255, 255, 230))
            self.ekf_lgv_quality_text.setZValue(21)
            # Counter-act the view's Y-flip so text reads normally
            self.ekf_lgv_quality_text.setTransform(QTransform.fromScale(1, -1))
            self.scene.addItem(self.ekf_lgv_quality_text)
        self.ekf_lgv_quality_text.setText(quality_pct)
        br = self.ekf_lgv_quality_text.boundingRect()
        # Centre horizontally; place inside the arrow body (~200 mm above the origin)
        self.ekf_lgv_quality_text.setPos(x_mm - br.width() / 2, y_mm + 200)

        if self._ekf_follow:
            self.centerOn(x_mm, y_mm)

        if self._ekf_zoom_on_first:
            self._ekf_zoom_on_first = False
            # Show a ~20 m × 20 m window centred on the LGV
            zoom_half = 10000.0   # mm
            from PyQt5.QtCore import QRectF
            self.fitInView(
                QRectF(x_mm - zoom_half, y_mm - zoom_half, zoom_half * 2, zoom_half * 2),
                Qt.KeepAspectRatio
            )
            self.ensure_y_flip()
            self.centerOn(x_mm, y_mm)

    def update_associated_reflectors(self, reflectors_mm):
        """Update green associated-reflector dots using an item pool to avoid scene invalidations."""
        needed = len(reflectors_mm)
        r = self.reflector_radius * 4
        ring_width = max(r * 0.45, 8)
        green_pen = QPen(QColor(0, 220, 80, 230), ring_width)
        label_font = QFont("Arial", max(1, int(r * 0.75)))
        flip_tf = QTransform.fromScale(1, -1)

        # Shrink pool: remove excess items from scene (pop label first, then ellipse)
        while len(self.ekf_associated_items) // 2 > needed:
            self.scene.removeItem(self.ekf_associated_items.pop())  # label
            self.scene.removeItem(self.ekf_associated_items.pop())  # ellipse

        # Grow pool: create new items only when count increases
        while len(self.ekf_associated_items) // 2 < needed:
            ellipse = QGraphicsEllipseItem(0, 0, 1, 1)
            ellipse.setBrush(Qt.transparent)
            ellipse.setPen(green_pen)
            ellipse.setZValue(5)
            self.scene.addItem(ellipse)
            label = QGraphicsSimpleTextItem("")
            label.setFont(label_font)
            label.setBrush(QColor(255, 255, 100))
            label.setTransform(flip_tf)
            label.setZValue(6)
            self.scene.addItem(label)
            self.ekf_associated_items.append(ellipse)
            self.ekf_associated_items.append(label)

        # Update all active items in-place (no add/remove = no scene invalidation)
        for i, (x, y, lid) in enumerate(reflectors_mm):
            ellipse = self.ekf_associated_items[i * 2]
            ellipse.setPen(green_pen)
            ellipse.setRect(x - r, y - r, r * 2, r * 2)
            label = self.ekf_associated_items[i * 2 + 1]
            label.setText(str(lid))
            label.setPos(x + r * 1.15, y + r * 0.5)

    def add_unassociated_reflectors(self, reflectors_mm, lgv_x_mm, lgv_y_mm):
        """Accumulate new non-associated reflector dots (EKF viewer — same size as green donuts)."""
        r = self.reflector_radius * 4   # match green donut radius
        red_color = QColor(255, 0, 0, 180)
        dot_pen = QPen(red_color)
        dot_pen.setWidth(0)
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        for x, y in reflectors_mm:
            ellipse = QGraphicsEllipseItem(
                x - r, y - r, r * 2, r * 2
            )
            ellipse.setPen(dot_pen)
            ellipse.setBrush(red_color)
            ellipse.setZValue(1)
            self.scene.addItem(ellipse)
            self.ekf_unassociated_items.append(ellipse)
            self.ekf_unassociated_data.append(
                (timestamp, round(x), round(y), round(lgv_x_mm), round(lgv_y_mm))
            )

    def clear_ekf_unassociated(self):
        """Remove only the accumulated non-associated reflector dots from the scene."""
        for item in self.ekf_unassociated_items:
            self.scene.removeItem(item)
        self.ekf_unassociated_items.clear()
        self.ekf_unassociated_data.clear()

    def clear_ekf_items(self):
        """Remove all EKF viewer visual elements and reset EKF state."""
        if self.ekf_lgv_arrow is not None:
            self.scene.removeItem(self.ekf_lgv_arrow)
            self.ekf_lgv_arrow = None
        if self.ekf_lgv_quality_text is not None:
            self.scene.removeItem(self.ekf_lgv_quality_text)
            self.ekf_lgv_quality_text = None
        for item in self.ekf_associated_items:
            self.scene.removeItem(item)
        self.ekf_associated_items.clear()
        for item in self.ekf_unassociated_items:
            self.scene.removeItem(item)
        self.ekf_unassociated_items.clear()
        self.ekf_unassociated_data.clear()
        self._ekf_follow = False
        self._ekf_zoom_on_first = False

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

    ekf_viewer = ET.SubElement(root, "EKF_Viewer")
    ekf_conn = ET.SubElement(ekf_viewer, "Connection")
    ET.SubElement(ekf_conn, "AmsNetId").text = '192.168.11.2.1.1'
    ET.SubElement(ekf_conn, "Port").text = '851'
    ET.SubElement(ekf_conn, "ReadIntervalMs").text = '1000'

    ekf_syms_root = ET.SubElement(ekf_viewer, "Symbols")
    _ekf_sym_defs = [
        ("AvoidReflectorCheck", "CustomPlcAttribute.AvoidReflectorCheck_sp", True),
        ("Quality",             "Sys_ExternalLocalization.extPoseInfo.quality", True),
        ("Aut_Run",             "LibraryInterfaces.LGV.Status.Aut_Run", True),
        ("Man_Run",             "LibraryInterfaces.LGV.Status.Man_Run", True),
        ("IsNotMoving",         "LibraryInterfaces.LGV.Status.IsNotMoving", True),
        ("LgvPosX",             "LibraryInterfaces.LGV.Guid.Info.Pos.X", False),
        ("LgvPosY",             "LibraryInterfaces.LGV.Guid.Info.Pos.Y", False),
        ("LgvPosH",             "LibraryInterfaces.LGV.Guid.Info.Pos.H", False),
        ("NumLGV",              "LibraryInterfaces.LGV.Info.NumLGV", False),
        ("ForwMotion",          "LibraryInterfaces.LGV.Guid.Rout.Cur_Seg_Info.Forw", False),
        ("Reflectors",          "Sys_ExternalLocalization.extReflectorSet[1].reflectors", False),
    ]
    for _sname, _spath, _has_bypass in _ekf_sym_defs:
        _selem = ET.SubElement(ekf_syms_root, _sname)
        ET.SubElement(_selem, "Symbol").text = _spath
        if _has_bypass:
            ET.SubElement(_selem, "Bypass").text = 'False'

    ET.indent(root, space="  ", level=0)

    config_path = get_config_path()
    tree = ET.ElementTree(root)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated config file at: {config_path}")

_REQUIRED_CONFIG_PATHS = [
    "DBSCAN/EPS",
    "DBSCAN/MinSamples",
    "ConfidenceScores/ConfidenceCheck",
    "ConfidenceScores/MinConfidence",
    "ConfidenceScores/FrequencyScore/Weight",
    "ConfidenceScores/FrequencyScore/MaxFreqThreshold",
    "ConfidenceScores/LGVDiversityScore/Weight",
    "ConfidenceScores/LGVDiversityScore/MaxLGVThreshold",
    "ConfidenceScores/TimestampScore/Weight",
    "ConfidenceScores/TimestampScore/MaxTimeVarianceSec",
    "ConfidenceScores/SpatialDistributionScore/Weight",
    "ConfidenceScores/SpatialDistributionScore/MaxSpatialStdDev",
    "GUI_Settings/Logged_Points/PointRadius",
    "GUI_Settings/DB3_Reflector/PointRadius",
    "GUI_Settings/Found_Reflector/PointRadius",
    "GUI_Settings/Found_Reflector/HighlightRadius",
    "EKF_Viewer/Connection/AmsNetId",
    "EKF_Viewer/Connection/Port",
    "EKF_Viewer/Connection/ReadIntervalMs",
    "EKF_Viewer/Symbols",
]

def is_config_compatible(root):
    """Return True if all required XML elements are present in the config."""
    for path in _REQUIRED_CONFIG_PATHS:
        if root.find(path) is None:
            print(f"[WARNING] Missing config element: {path}")
            return False
    return True


def load_configuration():
    """
    Load configuration from XML file. Generates default config if file doesn't exist.
    Returns:
        tuple: (config_created, eps, min_samples, confidence_check, min_confidence, ...)
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
    else:
        try:
            _tree = ET.parse(config_path)
            if not is_config_compatible(_tree.getroot()):
                print("[WARNING] Configuration file is incompatible. Regenerating...")
                generate_configfile()
                config_created = True
        except ET.ParseError:
            print("[WARNING] Configuration file is malformed. Regenerating...")
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

    # --- EKF Viewer config ---
    ekf_section = root.find("EKF_Viewer")
    if ekf_section is not None:
        ekf_cfg = {
            "ams_net_id": ekf_section.find("Connection/AmsNetId").text,
            "port": int(ekf_section.find("Connection/Port").text),
            "read_interval_ms": int(ekf_section.find("Connection/ReadIntervalMs").text),
            "symbols": {},
        }
        for sym_elem in ekf_section.find("Symbols"):
            sym_name = sym_elem.tag
            sym_path = sym_elem.find("Symbol").text
            bypass_elem = sym_elem.find("Bypass")
            bypass = bypass_elem is not None and bypass_elem.text.lower() == 'true'
            ekf_cfg["symbols"][sym_name] = (sym_path, bypass)
    else:
        ekf_cfg = {
            "ams_net_id": "192.168.11.2.1.1",
            "port": 851,
            "read_interval_ms": 1000,
            "symbols": {
                "AvoidReflectorCheck": ("CustomPlcAttribute.AvoidReflectorCheck_sp", False),
                "Quality":             ("Sys_ExternalLocalization.extPoseInfo.quality", False),
                "Aut_Run":             ("LibraryInterfaces.LGV.Status.Aut_Run", False),
                "Man_Run":             ("LibraryInterfaces.LGV.Status.Man_Run", False),
                "IsNotMoving":         ("LibraryInterfaces.LGV.Status.IsNotMoving", False),
                "LgvPosX":             ("LibraryInterfaces.LGV.Guid.Info.Pos.X", False),
                "LgvPosY":             ("LibraryInterfaces.LGV.Guid.Info.Pos.Y", False),
                "LgvPosH":             ("LibraryInterfaces.LGV.Guid.Info.Pos.H", False),
                "NumLGV":              ("LibraryInterfaces.LGV.Info.NumLGV", False),
                "ForwMotion":          ("LibraryInterfaces.LGV.Guid.Rout.Cur_Seg_Info.Forw", False),
                "Reflectors":          ("Sys_ExternalLocalization.extReflectorSet[1].reflectors", False),
            },
        }

    return (config_created, eps, min_samples, confidence_check, min_confidence,
            freq_weight, max_freq_threshold, lgv_div_weight, max_lgv_threshold,
            timestamp_weight, max_time_variance, spatial_weight, max_spatial_stddev,
            log_radius, db3_radius, reflector_radius, highlight_radius, ekf_cfg)


_EKF_BYPASS_SYMBOL_NAMES = {"AvoidReflectorCheck", "Quality", "Aut_Run", "Man_Run", "IsNotMoving"}


def save_ekf_config(ekf_config):
    """Update only the EKF_Viewer section of the config XML without touching other settings."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        return
    tree = ET.parse(config_path)
    root = tree.getroot()

    existing = root.find("EKF_Viewer")
    if existing is not None:
        root.remove(existing)

    ekv = ET.SubElement(root, "EKF_Viewer")
    conn = ET.SubElement(ekv, "Connection")
    ET.SubElement(conn, "AmsNetId").text = ekf_config["ams_net_id"]
    ET.SubElement(conn, "Port").text = str(ekf_config["port"])
    ET.SubElement(conn, "ReadIntervalMs").text = str(ekf_config["read_interval_ms"])

    syms_elem = ET.SubElement(ekv, "Symbols")
    for sym_name, (sym_path, bypass) in ekf_config["symbols"].items():
        sym_e = ET.SubElement(syms_elem, sym_name)
        ET.SubElement(sym_e, "Symbol").text = sym_path
        if sym_name in _EKF_BYPASS_SYMBOL_NAMES:
            ET.SubElement(sym_e, "Bypass").text = str(bypass)

    ET.indent(root, space="  ", level=0)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print("[INFO] EKF Viewer config saved.")


def save_analysis_config(cfg):
    """Update only the DBSCAN and ConfidenceScores sections of the config XML."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        return
    tree = ET.parse(config_path)
    root = tree.getroot()

    root.find("DBSCAN/EPS").text                                              = str(cfg["eps"])
    root.find("DBSCAN/MinSamples").text                                       = str(cfg["min_samples"])
    root.find("ConfidenceScores/ConfidenceCheck").text                        = str(cfg["confidence_check"])
    root.find("ConfidenceScores/MinConfidence").text                          = str(cfg["min_confidence"])
    root.find("ConfidenceScores/FrequencyScore/Weight").text                  = str(cfg["freq_weight"])
    root.find("ConfidenceScores/FrequencyScore/MaxFreqThreshold").text        = str(cfg["max_freq_threshold"])
    root.find("ConfidenceScores/LGVDiversityScore/Weight").text               = str(cfg["lgv_div_weight"])
    root.find("ConfidenceScores/LGVDiversityScore/MaxLGVThreshold").text      = str(cfg["max_lgv_threshold"])
    root.find("ConfidenceScores/TimestampScore/Weight").text                  = str(cfg["timestamp_weight"])
    root.find("ConfidenceScores/TimestampScore/MaxTimeVarianceSec").text      = str(cfg["max_time_variance"])
    root.find("ConfidenceScores/SpatialDistributionScore/Weight").text        = str(cfg["spatial_weight"])
    root.find("ConfidenceScores/SpatialDistributionScore/MaxSpatialStdDev").text = str(cfg["max_spatial_stddev"])

    ET.indent(root, space="  ", level=0)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print("[INFO] Analysis config saved.")


def save_visualization_config(cfg):
    """Update only the GUI_Settings section of the config XML."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        return
    tree = ET.parse(config_path)
    root = tree.getroot()

    root.find("GUI_Settings/Logged_Points/PointRadius").text       = str(cfg["log_radius"])
    root.find("GUI_Settings/DB3_Reflector/PointRadius").text       = str(cfg["db3_radius"])
    root.find("GUI_Settings/Found_Reflector/PointRadius").text     = str(cfg["reflector_radius"])
    root.find("GUI_Settings/Found_Reflector/HighlightRadius").text = str(cfg["highlight_radius"])

    ET.indent(root, space="  ", level=0)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print("[INFO] Visualization config saved.")


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
        self.console.setMaximumHeight(500)  # Limit height
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

        # EKF Viewer runtime state
        self._ekf_thread = None
        self._ekf_lgv_number = 0
        
        self.destroyed.connect(QApplication.quit)

    def init_menu(self):
        """Initialize menu bar with basic functionality"""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("File")

        #Load layout db3 action
        load_layout_action = QAction("Load Layout", self)
        load_layout_action.triggered.connect(self.load_layout_file)
        file_menu.addAction(load_layout_action)

        # Load unassociated reflectors CSV (TC3 format)
        load_csv_tc3_action = QAction("Load Non-Associated Logs", self)
        load_csv_tc3_action.triggered.connect(self.load_csv_files_tc3)
        file_menu.addAction(load_csv_tc3_action)

        # Load associated reflectors IDs log
        load_assoc_action = QAction("Load Associated Logs", self)
        load_assoc_action.triggered.connect(self.load_associated_ids)
        file_menu.addAction(load_assoc_action)

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
        clear_layout_action = QAction("Clear Layout", self)
        clear_layout_action.triggered.connect(self.viewer.clear_layout)
        file_menu.addAction(clear_layout_action)

        # Clear db3 reflectors only action
        clear_layout_reflectors_action = QAction("Clear DB3 Reflectors", self)
        clear_layout_reflectors_action.triggered.connect(self.viewer.clear_layout_reflectors)
        file_menu.addAction(clear_layout_reflectors_action)
        
        file_menu.addSeparator()
        
        # Clear console action
        clear_console_action = QAction("Clear Console", self)
        clear_console_action.triggered.connect(self.clear_console)
        file_menu.addAction(clear_console_action)

        # EDIT MENU
        edit_menu = menubar.addMenu("Edit")

        vis_settings_action = QAction("Visualization Settings", self)
        vis_settings_action.triggered.connect(self.open_visualization_settings)
        edit_menu.addAction(vis_settings_action)

        # ANALYSIS MENU
        analysis_menu = menubar.addMenu("Analysis")
        
        # Find TC3 Reflectors
        find_tc3_reflectors_action = QAction("Find Reflectors", self)
        find_tc3_reflectors_action.triggered.connect(self.reload_configuration)
        find_tc3_reflectors_action.triggered.connect(self.find_reflectors_placeholder)
        analysis_menu.addAction(find_tc3_reflectors_action)

        # Check not-used reflectors against loaded associated IDs
        check_unused_action = QAction("Check Not-Used Reflectors", self)
        check_unused_action.triggered.connect(self.check_not_used_reflectors)
        analysis_menu.addAction(check_unused_action)

        # Analysis Settings
        analysis_settings_action = QAction("Analysis Settings", self)
        analysis_settings_action.triggered.connect(self.open_analysis_settings)
        analysis_menu.addAction(analysis_settings_action)

        # Reload configuration action
        reload_config_action = QAction("Reload Configuration", self)
        reload_config_action.triggered.connect(self.reload_configuration)
        analysis_menu.addAction(reload_config_action)

        # EKF VIEWER MENU
        ekf_menu = menubar.addMenu("EKF Viewer")

        self._ekf_action_start = QAction("Start", self)
        self._ekf_action_start.triggered.connect(self.start_ekf_viewer)
        ekf_menu.addAction(self._ekf_action_start)

        self._ekf_action_stop = QAction("Stop", self)
        self._ekf_action_stop.triggered.connect(self.stop_ekf_viewer)
        self._ekf_action_stop.setEnabled(False)
        ekf_menu.addAction(self._ekf_action_stop)

        ekf_menu.addSeparator()

        self._ekf_action_save = QAction("Save Non-Associated...", self)
        self._ekf_action_save.triggered.connect(self.save_ekf_unassociated_csv)
        self._ekf_action_save.setEnabled(False)
        ekf_menu.addAction(self._ekf_action_save)

        self._ekf_action_clear_unassoc = QAction("Clear Non-Associated Points", self)
        self._ekf_action_clear_unassoc.triggered.connect(self._clear_ekf_unassoc_handler)
        self._ekf_action_clear_unassoc.setEnabled(False)
        ekf_menu.addAction(self._ekf_action_clear_unassoc)

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
            log_radius, db3_radius, reflector_radius, highlight_radius, ekf_cfg = load_configuration()
            
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
            globals()['ekf_config'] = ekf_cfg

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

    def load_associated_ids(self):
        """Load one or more associated-reflectors CSV files (column: ReflectorID) and
        accumulate the IDs into viewer.associated_reflector_ids."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Associated Reflectors Log(s)", "", "CSV Files (*.csv)"
        )
        if not file_paths:
            return

        before = len(self.viewer.associated_reflector_ids)
        for file_path in file_paths:
            try:
                with open(file_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'ReflectorID' in row:
                            try:
                                self.viewer.associated_reflector_ids.add(int(row['ReflectorID']))
                            except (ValueError, KeyError):
                                pass
            except Exception as e:
                self.log_to_console(f"[ERROR] Could not read {os.path.basename(file_path)}: {e}")

        added = len(self.viewer.associated_reflector_ids) - before
        msg = (
            f"[INFO] Associated IDs loaded: {added} new, "
            f"{len(self.viewer.associated_reflector_ids)} total unique ID(s) from "
            f"{len(file_paths)} file(s)."
        )
        print(msg)
        self.log_to_console(msg)

    def check_not_used_reflectors(self):
        """Compare layout reflectors against loaded associated IDs and highlight those never seen."""
        from ReflectorFinder import findUnusedReflectors

        if not self.viewer.layout_reflectors:
            self.log_to_console("[ERROR] No layout loaded. Please load a db3 layout first.")
            return
        if not self.viewer.associated_reflector_ids:
            self.log_to_console(
                "[ERROR] No associated IDs loaded. "
                "Use File → Load Associated Logs to load a Logger associated-reflectors file."
            )
            return

        unused = findUnusedReflectors(
            self.viewer.associated_reflector_ids,
            self.viewer.layout_reflectors
        )

        self.log_to_console(
            f"\nNEVER-SEEN REFLECTORS  ({len(unused)} / {len(self.viewer.layout_reflectors)} in layout)"
        )
        self.log_to_console("=" * 55)

        if unused:
            for rid, x, y in unused:
                self.log_to_console(f"  ID {rid:>6}  X={x:>10.1f}  Y={y:>10.1f}")
            self.log_to_console(f"\nTotal: {len(unused)} reflector(s) never seen.")
            self.viewer.visualize_unused_reflectors(unused)
        else:
            self.log_to_console("  All layout reflectors were seen at least once.")

        self.log_to_console("=" * 55)

    def open_analysis_settings(self):
        """Open the Analysis Settings dialog and persist any changes."""
        cfg = {
            "eps":               globals()['eps'],
            "min_samples":       globals()['min_samples'],
            "confidence_check":  globals()['confidence_check'],
            "min_confidence":    globals()['min_confidence'],
            "freq_weight":       globals()['freq_weight'],
            "max_freq_threshold":globals()['max_freq_threshold'],
            "lgv_div_weight":    globals()['lgv_div_weight'],
            "max_lgv_threshold": globals()['max_lgv_threshold'],
            "timestamp_weight":  globals()['timestamp_weight'],
            "max_time_variance": globals()['max_time_variance'],
            "spatial_weight":    globals()['spatial_weight'],
            "max_spatial_stddev":globals()['max_spatial_stddev'],
        }
        dlg = AnalysisSettingsDialog(cfg, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        new_cfg = dlg.get_config()
        save_analysis_config(new_cfg)
        for key, val in new_cfg.items():
            globals()[key] = val
        self.log_to_console("[INFO] Analysis settings saved.")

    def open_visualization_settings(self):
        """Open the Visualization Settings dialog and persist any changes."""
        old_cfg = {
            "log_radius":       globals()['log_radius'],
            "db3_radius":       globals()['db3_radius'],
            "reflector_radius": globals()['reflector_radius'],
            "highlight_radius": globals()['highlight_radius'],
        }
        dlg = VisualizationSettingsDialog(old_cfg, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        new_cfg = dlg.get_config()
        if new_cfg == old_cfg:
            return
        save_visualization_config(new_cfg)
        globals()['log_radius']       = new_cfg["log_radius"]
        globals()['db3_radius']       = new_cfg["db3_radius"]
        globals()['reflector_radius'] = new_cfg["reflector_radius"]
        globals()['highlight_radius'] = new_cfg["highlight_radius"]
        self.viewer.log_radius        = new_cfg["log_radius"]
        self.viewer.db3_radius        = new_cfg["db3_radius"]
        self.viewer.reflector_radius  = new_cfg["reflector_radius"]
        self.viewer.highlight_radius  = new_cfg["highlight_radius"]
        self.viewer.apply_visualization_resize(old_cfg, new_cfg)
        self.log_to_console("[INFO] Visualization settings saved.")
    
    def load_layout_file(self):
        """Load layout reflectors and background geometry from db3."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Layout.db3", "", "DB3 Files (*.db3)")
        if file_path:
            self.viewer.load_layout(file_path)
    
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
            confidence_threshold=min_confidence if confidence_check else 0.0,
            offset_correction=False,
            eps =  globals()['eps'],
            min_samples = globals()['min_samples'],

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
    # ------------------------------------------------------------------
    # EKF Viewer — MainWindow methods
    # ------------------------------------------------------------------

    def start_ekf_viewer(self):
        """Open settings dialog and (re-)start the EKF reader thread."""
        global ekf_config
        dlg = EKFSettingsDialog(ekf_config, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        new_cfg = dlg.get_config()
        if dlg.config_changed():
            save_ekf_config(new_cfg)
            ekf_config = new_cfg

        # Clear any existing EKF visuals (keep map/log dots)
        self.viewer.clear_ekf_items()
        self.viewer.ensure_y_flip()
        self.viewer.set_ekf_follow(True)
        self.viewer._ekf_zoom_on_first = True

        self._ekf_thread = EKFReaderThread(new_cfg)
        self._ekf_thread.cycle_data.connect(self.on_ekf_cycle)
        self._ekf_thread.lgv_number.connect(lambda n: setattr(self, '_ekf_lgv_number', n))
        self._ekf_thread.connection_status.connect(self.log_to_console)
        self._ekf_thread.finished.connect(self.on_ekf_stopped)
        self._ekf_thread.start()

        self._ekf_action_start.setEnabled(False)
        self._ekf_action_stop.setEnabled(True)
        self.log_to_console("[INFO] EKF Viewer starting...")

    def on_ekf_cycle(self, lgv_x_mm, lgv_y_mm, lgv_h_cdeg, associated_m, new_unassoc_m, conditions_met, forw_motion, quality):
        """Handle one EKF data cycle from the reader thread (runs in GUI thread via signal).

        lgv_x_mm / lgv_y_mm : LGV position already in mm (TC3 Pos.X/Y PLC symbol).
        associated_m / new_unassoc_m : reflector worldX/Y in metres → convert ×1000 to mm.
        """
        refl_scale = 1000.0   # reflector world coords: metres → mm
        assoc_mm   = [(x * refl_scale, y * refl_scale, lid) for x, y, lid in associated_m]
        unassoc_mm = [(x * refl_scale, y * refl_scale) for x, y in new_unassoc_m]

        self.viewer.update_lgv_arrow(lgv_x_mm, lgv_y_mm, lgv_h_cdeg, forw_motion, quality)
        self.viewer.update_associated_reflectors(assoc_mm)
        if unassoc_mm:
            self.viewer.add_unassociated_reflectors(unassoc_mm, lgv_x_mm, lgv_y_mm)
            self._ekf_action_save.setEnabled(True)
            self._ekf_action_clear_unassoc.setEnabled(True)

    def stop_ekf_viewer(self):
        """Request the EKF reader thread to stop gracefully."""
        if self._ekf_thread is not None:
            self._ekf_thread.stop()
        self.viewer.set_ekf_follow(False)
        self._ekf_action_stop.setEnabled(False)
        self.log_to_console("[INFO] EKF Viewer stopping...")

    def on_ekf_stopped(self):
        """Called when the EKF reader thread has fully finished."""
        self._ekf_thread = None
        self._ekf_action_start.setEnabled(True)
        self._ekf_action_stop.setEnabled(False)
        self.log_to_console("[INFO] EKF Viewer stopped.")

    def save_ekf_unassociated_csv(self):
        """Save accumulated non-associated reflector data to a CSV file."""
        data = self.viewer.ekf_unassociated_data
        if not data:
            self.log_to_console("[WARNING] No non-associated data to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Non-Associated Reflectors", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Lgv", "Timestamp", "WorldX", "WorldY", "LgvX", "LgvY"])
                for timestamp, x_mm, y_mm, lgv_x_mm, lgv_y_mm in data:
                    writer.writerow([self._ekf_lgv_number, timestamp, x_mm, y_mm, lgv_x_mm, lgv_y_mm])
            self.log_to_console(f"[INFO] Saved {len(data)} non-associated reflector(s) to: {file_path}")
        except Exception as e:
            self.log_to_console(f"[ERROR] Failed to save CSV: {e}")

    def _clear_ekf_unassoc_handler(self):
        """Clear accumulated non-associated EKF dots and reset related actions."""
        self.viewer.clear_ekf_unassociated()
        self._ekf_action_clear_unassoc.setEnabled(False)
        self._ekf_action_save.setEnabled(False)
        self.log_to_console("[INFO] Non-associated EKF points cleared.")

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
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    _base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    _icon = QIcon(os.path.join(_base, 'Icon.ico'))
    app.setWindowIcon(_icon)

    # --- Splash screen ---
    _splash_pix = QPixmap(os.path.join(_base, 'Icon.ico')).scaled(
        300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    _splash = QSplashScreen(_splash_pix, Qt.WindowStaysOnTopHint)
    _splash.show()
    _splash.showMessage("Loading configuration...", Qt.AlignBottom | Qt.AlignHCenter, QColor(200, 200, 200))
    app.processEvents()

    config_created, eps, min_samples, confidence_check, min_confidence, freq_weight, max_freq_threshold, \
    lgv_div_weight, max_lgv_threshold, timestamp_weight, max_time_variance, spatial_weight, max_spatial_stddev, \
    log_radius, db3_radius, reflector_radius, highlight_radius, ekf_config = load_configuration()

    _splash.showMessage("Starting application...", Qt.AlignBottom | Qt.AlignHCenter, QColor(200, 200, 200))
    app.processEvents()

    window = MainWindow()
    window.setWindowIcon(_icon)
    window.setMinimumSize(1024, 600)
    window.showMaximized()

    _splash.finish(window)
    sys.exit(app.exec_())