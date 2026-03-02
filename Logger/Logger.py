#TC2 Sys_AGV_IsNotMoving  <<<< SAFECTRLLOGICIN.Sys_AGV_IsNotMoving ???
#Sys_Aut_Run Sys_MAN_Run Sys_Navigator_Info.Quality

import pyads
import ctypes
from types import SimpleNamespace
import time
import csv
import os
from datetime import datetime
import glob
import re
import xml.etree.ElementTree as ET
import sys
import math

NumReflectors = 50 #TC3
MAX_MEASURED_REFLECTORS = 100 #TC2

######################################################
###################TC3 Structs########################
######################################################
class ReflectorObs(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("rho", ctypes.c_float),
        ("phi", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("radius", ctypes.c_float),
        ("quality", ctypes.c_float),
    ]


class ReflectorLandmark(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int32),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("radius", ctypes.c_float),
        ("hMin", ctypes.c_float),
        ("hMax", ctypes.c_float),
    ]


class ReflectorInfo(ctypes.Structure):
    _fields_ = [
        ("obs", ReflectorObs),
        ("landmark", ReflectorLandmark),
        ("wrtAgvX", ctypes.c_float),
        ("wrtAgvY", ctypes.c_float),
        ("worldX_preUpd", ctypes.c_float),
        ("worldY_preUpd", ctypes.c_float),
        ("worldX", ctypes.c_float),
        ("worldY", ctypes.c_float),
        ("updateLag", ctypes.c_float),
        ("associated", ctypes.c_uint8),
        ("pad", ctypes.c_byte * 3),
    ]

######################################################
###################TC2 Structs########################
######################################################
class TIMESTRUCT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wYear", ctypes.c_uint16),
        ("wMonth", ctypes.c_uint16),
        ("wDayOfWeek", ctypes.c_uint16),
        ("wDay", ctypes.c_uint16),
        ("wHour", ctypes.c_uint16),
        ("wMinute", ctypes.c_uint16),
        ("wSecond", ctypes.c_uint16),
        ("wMilliseconds", ctypes.c_uint16),
    ]

class Nav_Ref(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("TimeDiff_ms", ctypes.c_uint16),
        ("Nav_raw_x_local_mm", ctypes.c_float),
        ("Nav_raw_y_local_mm", ctypes.c_float),
        ("raw_x_local_mm", ctypes.c_float),
        ("raw_y_local_mm", ctypes.c_float),
        ("mod_x_local_mm", ctypes.c_float),
        ("mod_y_local_mm", ctypes.c_float),
        ("mean_echo", ctypes.c_uint16),
        ("spot_num", ctypes.c_uint16),
    ]

Nav_Ref_Array = Nav_Ref * (MAX_MEASURED_REFLECTORS)

class Nav_Ref_Set(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("enable", ctypes.c_uint8),
        ("RawDataAvailable", ctypes.c_uint8),
        ("Valid", ctypes.c_uint8),
        ("Timestamp", TIMESTRUCT),
        ("TimeDiff_ms", ctypes.c_uint16),
        ("Reflector_Num", ctypes.c_uint16),
        ("Ref", Nav_Ref_Array),
        ("Bypass_ParallaxCorrectiont_request", ctypes.c_uint8),
        ("Bypass_ParallaxCorrectiont_used", ctypes.c_uint8),
        ("SensId", ctypes.c_uint16),
    ]

#To compare associations
class Refl_Associations(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nAssociated", ctypes.c_uint16),  # UINT
        ("nCandidates", ctypes.c_uint16),  # UINT
        ("bearings", ctypes.c_float * MAX_MEASURED_REFLECTORS),          # REAL
        ("distances", ctypes.c_float * MAX_MEASURED_REFLECTORS),         # REAL
        ("reflectors_x", ctypes.c_float * MAX_MEASURED_REFLECTORS),      # REAL
        ("reflectors_y", ctypes.c_float * MAX_MEASURED_REFLECTORS),      # REAL
        ("associated_id_meas", ctypes.c_uint16 * MAX_MEASURED_REFLECTORS),  # UINT
        ("associated_id_refl", ctypes.c_uint16 * MAX_MEASURED_REFLECTORS),  # UINT
        ("reflector_error_mm", ctypes.c_float * MAX_MEASURED_REFLECTORS),   # REAL
        ("rho_error_mm", ctypes.c_float * MAX_MEASURED_REFLECTORS),         # REAL
        ("theta_error_rad", ctypes.c_float * MAX_MEASURED_REFLECTORS),      # REAL
        ("isAssociated", ctypes.c_uint8 * MAX_MEASURED_REFLECTORS),         # BOOL
    ]

class Agv_Pos(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("X", ctypes.c_double),  # LREAL = c_double
        ("Y", ctypes.c_double),
        ("H", ctypes.c_double),  # centidegrees
    ]


def is_tc2_environment():
    """
    Detect if running in a TC2 environment by checking for TC2-specific paths.
    
    Returns:
        bool: True if TC2 environment detected, False otherwise
    """
    # TC2 uses C:\Backup directory
    return os.path.exists(r'C:\Backup')


def get_log_path(lgvNum):
    """
    Generate a timestamped log file path for the given LGV.
    
    Args:
        lgvNum: LGV number (e.g., 5 for LGV5)
        
    Returns:
        Full path to the new log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"LGV{lgvNum}_MissingReflector_log_{timestamp}.csv"
    return os.path.join(log_dir, filename)


def get_log_folder_size_mb():
    """
    Calculate the total size of all CSV log files in the log directory.
    
    Returns:
        Total size in megabytes
    """
    total_size = 0
    for f in glob.glob(os.path.join(log_dir, "*.csv")):
        total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)


def cleanup_old_logs():
    """
    Delete oldest log files when total folder size exceeds MAX_LOG_FOLDER_SIZE_MB.
    Files are deleted in order from oldest to newest until size is under limit.
    """
    files = sorted(
        glob.glob(os.path.join(log_dir, "*.csv")),
        key=os.path.getmtime
    )
    while get_log_folder_size_mb() > MAX_LOG_FOLDER_SIZE_MB and files:
        oldest = files.pop(0)
        print(f"Deleting old log file: {oldest}")
        os.remove(oldest)


def extract_lgv_number(): 
    """
    Extract the LGV number from LGV.XML (TC3) or LGVxx.xml filename (TC2).
    
    Returns:
        LGV number as integer (e.g., 5 from "LGV5" or 57 from "LGV57.xml")
        Returns 0 if no LGV number is found
    """
    config_dir_tc3 = r'D:\Config'
    config_dir_tc2 = r'C:\Backup'
    
    # Try TC3 approach first: read content of LGV.XML
    lgv_xml_path = os.path.join(config_dir_tc3, 'LGV.XML')
    if os.path.exists(lgv_xml_path):
        try:
            with open(lgv_xml_path, 'r') as file:
                text = file.read().strip()
            
            match = re.search(r'LGV\s*(\d{1,2})', text)
            if match:
                extracted_number = int(match.group(1))
                print(f"Extracted LGV number from file content: {extracted_number}")
                return extracted_number
        except Exception as e:
            print(f"[WARNING] Error reading LGV.XML: {e}")
    
    # Try TC2 approach: search for LGVxx.xml filename
    if os.path.exists(config_dir_tc2):
        try:
            for filename in os.listdir(config_dir_tc2):
                if filename.upper().startswith('LGV') and filename.upper().endswith('.XML'):
                    match = re.search(r'LGV\s*(\d{1,2})\.xml', filename, re.IGNORECASE)
                    if match:
                        extracted_number = int(match.group(1))
                        print(f"Extracted LGV number from filename '{filename}': {extracted_number}")
                        return extracted_number
        except Exception as e:
            print(f"[WARNING] Error scanning config directory: {e}")
    
    print("[WARNING] No LGV number found in LGV.XML content or LGVxx.xml filename. Using 0.")
    return 0


def get_config_path():
    """
    Determine the best location for the config file.
    
    TC3: Tries D:\Apps\ReflectorFinder first, falls back to exe directory if D:\Apps doesn't exist.
    TC2: Uses C:\ReflectorFinder
    
    Returns:
        Path to the config file
    """
    # Check if TC2 environment
    if is_tc2_environment():
        tc2_path = r'C:\ReflectorFinder'
        os.makedirs(tc2_path, exist_ok=True)
        print(f"[INFO] TC2 environment detected. Using config path: {tc2_path}")
        return os.path.join(tc2_path, 'RFConfig.xml')
    
    # TC3 environment
    primary_path = r'D:\Apps\ReflectorFinder'
    
    # Check if D:\Apps exists
    if os.path.exists(r'D:\Apps'):
        os.makedirs(primary_path, exist_ok=True)
        return os.path.join(primary_path, 'RFConfig.xml')
    else:
        # Fallback to exe/script directory
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            exe_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            exe_dir = os.path.dirname(os.path.abspath(__file__))
        
        fallback_path = os.path.join(exe_dir, 'ReflectorFinder')
        os.makedirs(fallback_path, exist_ok=True)
        print(f"[WARNING] D:\\Apps not found. Using fallback location: {fallback_path}")
        return os.path.join(fallback_path, 'RFConfig.xml')


def generate_configfile():
    """
    Generate a default configuration XML file with standard settings.
    Creates RFConfig.xml with default PLC connection
    settings, logging parameters, and symbol definitions.
    
    Adjusts paths based on TC2/TC3 environment.
    """
    root = ET.Element("Configuration")
    comm = ET.SubElement(root, "Communication")
    ET.SubElement(comm, "AmsNetId").text = '192.168.11.2.1.1'
    ET.SubElement(comm, "Port").text = '851'

    log = ET.SubElement(root, "Log_config")
    ET.SubElement(log, "OneFile").text = 'True'
    ET.SubElement(log, "LogDurationHours").text = '8.0'
    ET.SubElement(log, "ReadIntervalMs").text = '1000'
    
    # Set log folder based on environment
    if is_tc2_environment():
        ET.SubElement(log, "LogFolder").text = r'C:\Backup\MissingReflectors'
        print("[INFO] TC2 environment detected. Using TC2 log path.")
    else:
        ET.SubElement(log, "LogFolder").text = r'D:\Logs\MissingReflectors'
    
    ET.SubElement(log, "MaxLogFolderSizeMB").text = '20'

    #==========TC3==========
    symbols_tc3 = ET.SubElement(root, "TC3_Symbols")
    
    avoid_reflector = ET.SubElement(symbols_tc3, "AvoidReflectorCheck")
    ET.SubElement(avoid_reflector, "Symbol").text = "CustomPlcAttribute.AvoidReflectorCheck_sp"
    ET.SubElement(avoid_reflector, "Bypass").text = 'False'

    quality = ET.SubElement(symbols_tc3, "Quality")
    ET.SubElement(quality, "Symbol").text = "Sys_ExternalLocalization.extPoseInfo.quality"
    ET.SubElement(quality, "Bypass").text = 'False'

    aut_run = ET.SubElement(symbols_tc3, "Aut_Run")
    ET.SubElement(aut_run, "Symbol").text = "LibraryInterfaces.LGV.Status.Aut_Run"
    ET.SubElement(aut_run, "Bypass").text = 'False'

    man_run = ET.SubElement(symbols_tc3, "Man_Run")
    ET.SubElement(man_run, "Symbol").text = "LibraryInterfaces.LGV.Status.Man_Run"
    ET.SubElement(man_run, "Bypass").text = 'False'

    isNotMoving = ET.SubElement(symbols_tc3, "IsNotMoving")
    ET.SubElement(isNotMoving, "Symbol").text = "LibraryInterfaces.LGV.Status.IsNotMoving"
    ET.SubElement(isNotMoving, "Bypass").text = 'False'

    lgvPosX = ET.SubElement(symbols_tc3, "LgvPosX")
    ET.SubElement(lgvPosX, "Symbol").text = "LibraryInterfaces.LGV.Guid.Info.Pos.X"

    lgvPosY = ET.SubElement(symbols_tc3, "LgvPosY")
    ET.SubElement(lgvPosY, "Symbol").text = "LibraryInterfaces.LGV.Guid.Info.Pos.Y"

    reflectors = ET.SubElement(symbols_tc3, "Reflectors")
    ET.SubElement(reflectors, "Symbol").text = "Sys_ExternalLocalization.extReflectorSet[1].reflectors"

    #==========TC2==========
    symbols_tc2 = ET.SubElement(root, "TC2_Symbols")
    
    ekf_reflector = ET.SubElement(symbols_tc2, "EKF_reflector")
    ET.SubElement(ekf_reflector, "Symbol").text = ".EKF_reflectors"

    assoc_reflector = ET.SubElement(symbols_tc2, "Associated_reflector")
    ET.SubElement(assoc_reflector, "Symbol").text = ".EKF_associations_PostUpdate"

    quality_tc2 = ET.SubElement(symbols_tc2, "Quality")
    ET.SubElement(quality_tc2, "Symbol").text = ".Sys_Navigator_Info.Quality"
    ET.SubElement(quality_tc2, "Bypass").text = 'False'

    aut_run_tc2 = ET.SubElement(symbols_tc2, "Aut_Run")
    ET.SubElement(aut_run_tc2, "Symbol").text = ".Sys_Aut_Run"
    ET.SubElement(aut_run_tc2, "Bypass").text = 'False'

    man_run_tc2 = ET.SubElement(symbols_tc2, "Man_Run")
    ET.SubElement(man_run_tc2, "Symbol").text = ".Sys_MAN_Run"
    ET.SubElement(man_run_tc2, "Bypass").text = 'False'

    isNotMoving_tc2 = ET.SubElement(symbols_tc2, "IsNotMoving")
    ET.SubElement(isNotMoving_tc2, "Symbol").text = ".Sys_AGV_IsNotMoving"
    ET.SubElement(isNotMoving_tc2, "Bypass").text = 'False'

    sys_agv_pos = ET.SubElement(symbols_tc2, "Sys_Agv_Pos")
    ET.SubElement(sys_agv_pos, "Symbol").text = ".Sys_Agv_Pos"

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
    config_path = get_config_path()
    config_created = False
    
    if not os.path.exists(config_path):
        print("[WARNING] Configuration file not found. Generating default config.")
        generate_configfile()
        config_created = True

    tree = ET.parse(config_path)
    root = tree.getroot()

    plc_ams_id = root.find("Communication/AmsNetId").text
    local_net_id = plc_ams_id.split('.')
    plc_ip = '.'.join(local_net_id[:4])
    port = int(root.find("Communication/Port").text)
    tc2 = False if port == 851 else True  # TC3 uses port 851, TC2 uses 801

    remoterun_enable = plc_ip is not None and plc_ip != '192.168.11.2'

    one_file = root.find("Log_config/OneFile").text.lower() == 'true'
    log_duration = float(root.find("Log_config/LogDurationHours").text)
    read_interval = int(root.find("Log_config/ReadIntervalMs").text)
    log_folder = root.find("Log_config/LogFolder").text
    max_log_size = int(root.find("Log_config/MaxLogFolderSizeMB").text)

    symbols_tc3 = {}
    for symbol in root.find("TC3_Symbols"):
        name = symbol.tag
        symb_text = symbol.find("Symbol").text
        bypass_elem = symbol.find("Bypass")
        bypass = bypass_elem is not None and bypass_elem.text.lower() == 'true'
        symbols_tc3[name] = (symb_text, bypass)
    
    symbols_tc2 = {}
    for symbol in root.find("TC2_Symbols"):
        name = symbol.tag
        symb_text = symbol.find("Symbol").text
        bypass_elem = symbol.find("Bypass")
        bypass = bypass_elem is not None and bypass_elem.text.lower() == 'true'
        symbols_tc2[name] = (symb_text, bypass)

    return (config_created, plc_ams_id, plc_ip, port, tc2, remoterun_enable, one_file, 
            log_duration, read_interval, log_folder, max_log_size, symbols_tc3, symbols_tc2)


def main():
    """
    Main logging loop that reads reflector data from PLC and logs unassociated reflectors.
    
    Continuously monitors PLC for unassociated reflectors when conditions are met:
    - AvoidReflectorCheck is not active (or bypassed)
    - LGV is in Auto or Manual run mode (or bypassed)
    - LGV is moving (or bypassed)
    - Localization quality is above 0.8 (or bypassed)
    
    Logs detected reflectors to CSV with timestamp and position information.
    """
    if one_file:
        if LOG_DURATION_HOURS == 0:
            print("[INFO] One file mode. Logging indefinitely.")
        else:
            print(f"[INFO] One file mode. Logging for {LOG_DURATION_HOURS} hours.")

    next_time = time.perf_counter()
    current_hour = datetime.now().hour
    log_file = open(get_log_path(lgv_num), mode='w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Lgv","Timestamp","WorldX", "WorldY", "LgvX", "LgvY"])

    start_time = time.perf_counter()
    end_time = start_time + (LOG_DURATION_HOURS * 3600) if LOG_DURATION_HOURS !=0 else None
    
    plc = pyads.Connection(PLC_AMS_ID, PORT, PLC_IP)
    plc.open()

    previousReflectors = []

    # Get PLC symbol handles
    if not tc2:
        avoid_reflector_bypass = symbols_tc3["AvoidReflectorCheck"][1]
        quality_bypass = symbols_tc3["Quality"][1]
        aut_run_bypass = symbols_tc3["Aut_Run"][1]
        man_run_bypass = symbols_tc3["Man_Run"][1]
        isNotMoving_bypass = symbols_tc3["IsNotMoving"][1]

        if not avoid_reflector_bypass: avoid_reflector_symbol = plc.get_symbol(symbols_tc3["AvoidReflectorCheck"][0])
        if not quality_bypass: quality_symbol = plc.get_symbol(symbols_tc3["Quality"][0])
        if not aut_run_bypass: aut_run_symbol = plc.get_symbol(symbols_tc3["Aut_Run"][0])
        if not man_run_bypass: man_run_symbol = plc.get_symbol(symbols_tc3["Man_Run"][0])
        if not isNotMoving_bypass: isNotMoving_symbol = plc.get_symbol(symbols_tc3["IsNotMoving"][0])
        LgvPosX_symbol = plc.get_symbol(symbols_tc3["LgvPosX"][0])
        LgvPosY_symbol = plc.get_symbol(symbols_tc3["LgvPosY"][0])
        #reflectors_symbol = plc.get_symbol(symbols_tc3["Reflectors"][0])
        
    
    else:
        quality_bypass = symbols_tc2["Quality"][1]
        aut_run_bypass = symbols_tc2["Aut_Run"][1]
        man_run_bypass = symbols_tc2["Man_Run"][1]
        isNotMoving_bypass = symbols_tc2["IsNotMoving"][1]

        avoid_reflector_bypass = True
        if not quality_bypass: quality_symbol = plc.get_symbol(symbols_tc2["Quality"][0])
        if not aut_run_bypass: aut_run_symbol = plc.get_symbol(symbols_tc2["Aut_Run"][0])
        if not man_run_bypass: man_run_symbol = plc.get_symbol(symbols_tc2["Man_Run"][0])
        if not isNotMoving_bypass: isNotMoving_symbol = plc.get_symbol(symbols_tc2["IsNotMoving"][0])
        
        ekf_reflector_symbol = symbols_tc2["EKF_reflector"][0]
        assoc_reflector_symbol = symbols_tc2["Associated_reflector"][0]
        sys_agv_pos_symbol = symbols_tc2["Sys_Agv_Pos"][0]

        size = ctypes.sizeof(Nav_Ref_Set)
        agv_pos_size = ctypes.sizeof(Agv_Pos)

        assoc_size = ctypes.sizeof(Refl_Associations)

    try:
        # Initialize variables
        avoid_reflector = False
        quality = 0.0
        aut_run = False
        man_run = False
        isNotMoving = True

        while True:
            # Check if duration has elapsed in one-file mode
            if end_time is not None and time.perf_counter() >= end_time:
                print("Reached max log duration. Stopping.")
                break

            # Create new hourly log file if hour has changed (multi-file mode only)
            now = datetime.now()
            if now.hour != current_hour and not one_file:
                log_file.close()
                cleanup_old_logs()
                log_file = open(get_log_path(lgv_num), mode='w', newline='')
                csv_writer = csv.writer(log_file)
                csv_writer.writerow(["Lgv","Timestamp","WorldX", "WorldY", "LgvX", "LgvY"])
                current_hour = now.hour
                print(f"Started new log file at {now.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # Read condition variables from PLC
                avoid_reflector = avoid_reflector_symbol.read() if not avoid_reflector_bypass else False
                quality = quality_symbol.read() if not quality_bypass else 0.0
                aut_run = aut_run_symbol.read() if not aut_run_bypass else False
                man_run = man_run_symbol.read() if not man_run_bypass else False
                isNotMoving = isNotMoving_symbol.read() if not isNotMoving_bypass else False

                # Evaluate conditions with bypass logic
                avoidref_ok = avoid_reflector_bypass or not avoid_reflector
                quality_ok = quality_bypass or quality > 0.8
                run_ok = (aut_run_bypass and man_run_bypass) or aut_run or man_run
                notmoving_ok = isNotMoving_bypass or not isNotMoving

                if avoidref_ok and run_ok and notmoving_ok and quality_ok:
                    print("Reading reflectors...")
                    if tc2:
                        #Raw reflectors
                        raw_data_list = plc.read_by_name(ekf_reflector_symbol, ctypes.c_ubyte * size)
                        raw_agv_pos = plc.read_by_name(sys_agv_pos_symbol, ctypes.c_ubyte * agv_pos_size)

                        #Associated reflectors
                        raw_assoc = plc.read_by_name(assoc_reflector_symbol, ctypes.c_ubyte * assoc_size)

                    else:
                        #raw_data_list = plc.read_by_name(
                        #    "Sys_ExternalLocalization.extReflectorSet[1].reflectors",
                        #    ctypes.c_ubyte * (ctypes.sizeof(ReflectorInfo) * NumReflectors)
                        #)
                        raw_data_list = plc.read_by_name(
                            symbols_tc3["Reflectors"][0],
                            ctypes.c_ubyte * (ctypes.sizeof(ReflectorInfo) * NumReflectors)
                        )
                else:
                    # Print specific reason for skipping
                    if not avoidref_ok:     print("AvoidReflectorCheck_sp active... skipping")
                    elif not run_ok:        print("LGV not in run... skipping")
                    elif not notmoving_ok:  print("LGV is not moving... skipping")
                    elif not quality_ok:    print(f'Low quality({quality:.3f})... skipping')
                    time.sleep(1)
                    continue

            
                    

            except pyads.ADSError as e:
                print(f"Read failed: {e}")
                try:
                    plc.close()
                except Exception:
                    pass
                time.sleep(1)
                try:
                    plc.open()
                    time.sleep(1)
                    continue
                except pyads.ADSError as conn_err:
                    print(f"Reconnection failed: {conn_err}")
                    time.sleep(1)
                    continue
            except Exception as unexpected:
                print(f"Unexpected error: {unexpected}")
                time.sleep(1)
                continue

            if not tc2:
                # Parse raw bytes into ReflectorInfo structs
                ReflectorArray = ReflectorInfo * NumReflectors
                reflectors = ReflectorArray.from_buffer_copy(bytes(raw_data_list))
            else:
                # TC2: Transform local reflector coordinates to global coordinates
                navset = Nav_Ref_Set.from_buffer_copy(bytes(raw_data_list))
                agv_pos = Agv_Pos.from_buffer_copy(bytes(raw_agv_pos))

                # TC2 coordinates: AGV position and reflector positions are in millimeters
                # Apply rotation matrix to transform from local (vehicle) to global (world) coordinates
                theta = math.radians(agv_pos.H * 0.01)  # Convert centidegrees to radians

                glob_reflectors = []
                for i in range(navset.Reflector_Num):
                    ref = navset.Ref[i]
                    # Rotation matrix: [cos(θ) -sin(θ)] applied to local coordinates
                    #                  [sin(θ)  cos(θ)]
                    rotated_x = ref.mod_x_local_mm * math.cos(theta) - ref.mod_y_local_mm * math.sin(theta)
                    rotated_y = ref.mod_x_local_mm * math.sin(theta) + ref.mod_y_local_mm * math.cos(theta)
                    glob_reflectors.append(SimpleNamespace(
                        worldX=agv_pos.X + rotated_x,
                        worldY=agv_pos.Y + rotated_y,
                        associated=False  # TC2: all reflectors treated as unassociated
                    ))

                #Get reflector associations and match with filtered reflectors. ONLY FOR TC2  
                associations = Refl_Associations.from_buffer_copy(bytes(raw_assoc))

                associated_meas_ids = set()
                for i in range(associations.nAssociated):
                    meas_id = associations.associated_id_meas[i] 
                    associated_meas_ids.add(meas_id)
                
                reflectors = []
                for i in range(navset.Reflector_Num):
                    if (i+1) not in associated_meas_ids:
                        reflectors.append(glob_reflectors[i])
                

            # Extract unassociated reflectors with non-zero world coordinates
            # Note: TC2 marks all reflectors as unassociated since association data is not available
            newReflectors = [
                (r.worldX, r.worldY, bool(r.associated))
                for r in reflectors
                if r.worldX != 0.0 and r.associated == False
            ]

            # Filter out reflectors that were already detected in previous cycle
            filteredReflectors = []
            for ref in newReflectors:
                if ref not in previousReflectors:
                    filteredReflectors.append(ref)
          

            # Log newly detected reflectors
            if filteredReflectors:
                print(f"Detected {len(filteredReflectors)} new unassociated reflectors:")

                if not tc2:
                    LgvPosX = round(LgvPosX_symbol.read())
                    LgvPosY = round(LgvPosY_symbol.read())
                else:
                    LgvPosX = round(agv_pos.X)
                    LgvPosY = round(agv_pos.Y)
                
                for i, (wx, wy, assoc) in enumerate(filteredReflectors):
                    timestamp = datetime.now().isoformat(timespec='milliseconds')
                    world_x_mm = round(wx) if tc2 else round(wx * 1000)
                    world_y_mm = round(wy) if tc2 else round(wy * 1000)
                    csv_writer.writerow([
                        lgv_num,
                        timestamp,
                        f"{world_x_mm}",
                        f"{world_y_mm}",
                        LgvPosX,
                        LgvPosY
                    ])
                    print(f"Reflector {i+1}: worldX={world_x_mm}, worldY={world_y_mm}")
                
                log_file.flush()

            previousReflectors = newReflectors

            # Maintain consistent read interval using precise timing
            next_time += VarReadInterval / 1000.0
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        plc.close()
        log_file.close()
        cleanup_old_logs()


if __name__ == "__main__":
    try:
        config_created, PLC_AMS_ID, PLC_IP, PORT, tc2, remoterun_enable, one_file, LOG_DURATION_HOURS, \
        VarReadInterval, log_dir, MAX_LOG_FOLDER_SIZE_MB, symbols_tc3, symbols_tc2 = load_configuration()

        if config_created:
            print("[INFO] Please review and adjust the generated configuration file as needed, then restart the logger.")
            sys.exit(0)

        # Override log directory if remote run is enabled
        if remoterun_enable:
            if getattr(sys, 'frozen', False):
                exe_dir = os.path.dirname(sys.executable)
            else:
                exe_dir = os.path.dirname(os.path.abspath(__file__))
            
            log_dir = os.path.join(exe_dir, 'ReflectorFinder', 'Logs')
            print(f"[INFO] Remote run mode: Logs will be saved to {log_dir}")
        else:
            # Check if log_dir drive exists, fallback to exe directory if not
            log_drive = os.path.splitdrive(log_dir)[0]
            if log_drive and not os.path.exists(log_drive + '\\'):
                if getattr(sys, 'frozen', False):
                    exe_dir = os.path.dirname(sys.executable)
                else:
                    exe_dir = os.path.dirname(os.path.abspath(__file__))
                
                log_dir = os.path.join(exe_dir, 'Logs', 'MissingReflectors')
                print(f"[WARNING] Configured log drive not found. Using fallback: {log_dir}")

        os.makedirs(log_dir, exist_ok=True)

        # Determine LGV number
        lgv_num = extract_lgv_number()
        #if not remoterun_enable:
        #    lgv_num = extract_lgv_number()
        #else:
        #    lgv_num = 0
        #    print("[INFO] Remote run mode enabled. Using LGV number 0.")

        main()

    except Exception as e:
        print("\n" + "="*50)
        print("FATAL ERROR - Logger crashed:")
        print("="*50)
        print(f"{type(e).__name__}: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("="*50)
    finally:
        input("\nPress Enter to exit...")