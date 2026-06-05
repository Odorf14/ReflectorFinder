import pyads
import ctypes
import time
import csv
import os
from datetime import datetime
import glob
import re
import xml.etree.ElementTree as ET
import sys

NumReflectors = 50

_REQUIRED_CONFIG_PATHS = [
    "Communication/AmsNetId",
    "Communication/Port",
    "Log_config/OneFile",
    "Log_config/LogDurationHours",
    "Log_config/ReadIntervalMs",
    "Log_config/LogFolder",
    "Log_config/MaxLogFolderSizeMB",
    "Log_config/LogMissingReflectors",
    "Log_config/LogAssociatedReflectors",
    "TC3_Symbols",
]

def is_config_compatible(root):
    """Return True if all required XML elements are present in the config."""
    for path in _REQUIRED_CONFIG_PATHS:
        if root.find(path) is None:
            print(f"[WARNING] Missing config element: {path}")
            return False
    return True

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



def get_unassociated_log_path(lgvNum):
    """
    Generate a timestamped log file path for unassociated (missing) reflectors.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"LGV{lgvNum}_UnassociatedReflectors_{timestamp}.csv"
    return os.path.join(log_dir, filename)


def get_associated_log_path(lgvNum):
    """
    Return the path to the cumulative associated reflector IDs log.
    This file is not timestamped — it accumulates IDs across sessions.
    """
    filename = f"LGV{lgvNum}_AssociatedReflectors.csv"
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
    Extract the LGV number from D:\\Config\\LGV.XML.

    Returns:
        LGV number as integer (e.g., 5 from "LGV5")
        Returns 0 if no LGV number is found
    """
    config_dir_tc3 = r'D:\Config'

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

    print("[WARNING] No LGV number found in LGV.XML. Using 0.")
    return 0


def get_config_path():
    """
    Determine the best location for the config file.
    Tries D:\\Apps\\ReflectorFinder first, falls back to the exe directory if D:\\Apps doesn't exist.

    Returns:
        Path to the config file
    """
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
    Creates RFConfig.xml with default PLC connection settings, logging parameters,
    and symbol definitions.
    """
    root = ET.Element("Configuration")
    comm = ET.SubElement(root, "Communication")
    ET.SubElement(comm, "AmsNetId").text = '192.168.11.2.1.1'
    ET.SubElement(comm, "Port").text = '851'

    log = ET.SubElement(root, "Log_config")
    ET.SubElement(log, "OneFile").text = 'True'
    ET.SubElement(log, "LogDurationHours").text = '8.0'
    ET.SubElement(log, "ReadIntervalMs").text = '1000'
    
    ET.SubElement(log, "LogFolder").text = r'D:\Logs\MissingReflectors'
    
    ET.SubElement(log, "MaxLogFolderSizeMB").text = '20'
    ET.SubElement(log, "LogMissingReflectors").text = 'True'
    ET.SubElement(log, "LogAssociatedReflectors").text = 'True'

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

    numLGV = ET.SubElement(symbols_tc3, "NumLGV")
    ET.SubElement(numLGV, "Symbol").text = "LibraryInterfaces.LGV.Info.NumLGV"

    reflectors = ET.SubElement(symbols_tc3, "Reflectors")
    ET.SubElement(reflectors, "Symbol").text = "Sys_ExternalLocalization.extReflectorSet[1].reflectors"

    ET.indent(root, space="  ", level=0)

    config_path = get_config_path()
    
    tree = ET.ElementTree(root)
    tree.write(config_path, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated config file at: {config_path}")


def load_configuration():
    """
    Load configuration from XML file. Generates default config if file doesn't exist.

    Returns:
        tuple: (config_created, plc_ams_id, plc_ip, port, remoterun_enable, one_file, log_duration, read_interval,
                log_folder, max_log_size, symbols_tc3, log_missing, log_associated)
    """
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

    plc_ams_id = root.find("Communication/AmsNetId").text
    local_net_id = plc_ams_id.split('.')
    plc_ip = '.'.join(local_net_id[:4])
    port = int(root.find("Communication/Port").text)
    remoterun_enable = plc_ip is not None and plc_ip != '192.168.11.2'

    one_file = root.find("Log_config/OneFile").text.lower() == 'true'
    log_duration = float(root.find("Log_config/LogDurationHours").text)
    read_interval = int(root.find("Log_config/ReadIntervalMs").text)
    log_folder = root.find("Log_config/LogFolder").text
    max_log_size = int(root.find("Log_config/MaxLogFolderSizeMB").text)
    log_missing    = root.find("Log_config/LogMissingReflectors").text.lower() == 'true'
    log_associated = root.find("Log_config/LogAssociatedReflectors").text.lower() == 'true'

    symbols_tc3 = {}
    for symbol in root.find("TC3_Symbols"):
        name = symbol.tag
        symb_text = symbol.find("Symbol").text
        bypass_elem = symbol.find("Bypass")
        bypass = bypass_elem is not None and bypass_elem.text.lower() == 'true'
        symbols_tc3[name] = (symb_text, bypass)
    
    return (config_created, plc_ams_id, plc_ip, port, remoterun_enable, one_file,
            log_duration, read_interval, log_folder, max_log_size, symbols_tc3,
            log_missing, log_associated)


def main():
    """
    Main logging loop that reads reflector data from PLC and logs unassociated
    and/or associated reflectors, depending on configuration flags.
    """
    global lgv_num

    if one_file:
        if LOG_DURATION_HOURS == 0:
            print("[INFO] One file mode. Logging indefinitely.")
        else:
            print(f"[INFO] One file mode. Logging for {LOG_DURATION_HOURS} hours.")

    print(f"[INFO] LogMissingReflectors={LOG_MISSING}  LogAssociatedReflectors={LOG_ASSOCIATED}")

    next_time = time.perf_counter()
    current_hour = datetime.now().hour

    # --- Unassociated (missing) reflectors log ---
    if LOG_MISSING:
        log_file = open(get_unassociated_log_path(lgv_num), mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["Lgv", "Timestamp", "WorldX", "WorldY", "LgvX", "LgvY"])
    else:
        log_file = None
        csv_writer = None

    # --- Associated reflectors log (cumulative, append) ---
    seen_associated_ids = set()
    assoc_log_file = None
    assoc_csv_writer = None
    if LOG_ASSOCIATED:
        assoc_path = get_associated_log_path(lgv_num)
        # Load already-seen IDs from file so we don't duplicate across sessions
        if os.path.exists(assoc_path):
            try:
                with open(assoc_path, 'r', newline='') as _f:
                    _reader = csv.DictReader(_f)
                    for _row in _reader:
                        try:
                            seen_associated_ids.add(int(_row['ReflectorID']))
                        except (ValueError, KeyError):
                            pass
                print(f"[INFO] Loaded {len(seen_associated_ids)} existing associated IDs from {assoc_path}")
            except Exception as e:
                print(f"[WARNING] Could not read existing associated log: {e}")
        assoc_log_file = open(assoc_path, 'a', newline='')
        assoc_csv_writer = csv.writer(assoc_log_file)
        if os.path.getsize(assoc_path) == 0:
            assoc_csv_writer.writerow(["ReflectorID"])

    start_time = time.perf_counter()
    end_time = start_time + (LOG_DURATION_HOURS * 3600) if LOG_DURATION_HOURS != 0 else None

    plc = pyads.Connection(PLC_AMS_ID, PORT, PLC_IP)
    plc.open()

    previousReflectors = []

    # Get PLC symbol handles
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
    # Cache reflectors symbol for direct index reads (avoids name resolution each cycle)
    _refl_sym = plc.get_symbol(symbols_tc3["Reflectors"][0])
    _refl_size = ctypes.sizeof(ReflectorInfo) * NumReflectors

    # Read LGV number directly from PLC (overrides extract_lgv_number)
    if "NumLGV" in symbols_tc3:
        try:
            lgv_num = int(plc.get_symbol(symbols_tc3["NumLGV"][0]).read())
            print(f"[INFO] LGV number from PLC: {lgv_num}")
        except Exception as e:
            print(f"[WARNING] Could not read NumLGV from PLC: {e}. Using {lgv_num}.")

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
                if log_file:
                    log_file.close()
                    cleanup_old_logs()
                    log_file = open(get_unassociated_log_path(lgv_num), mode='w', newline='')
                    csv_writer = csv.writer(log_file)
                    csv_writer.writerow(["Lgv", "Timestamp", "WorldX", "WorldY", "LgvX", "LgvY"])
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
                    raw_data_list = plc.read(
                        _refl_sym.index_group,
                        _refl_sym.index_offset,
                        ctypes.c_ubyte * _refl_size
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

            # Parse raw bytes into ReflectorInfo structs
            ReflectorArray = ReflectorInfo * NumReflectors
            reflectors = ReflectorArray.from_buffer_copy(bytes(raw_data_list))

            # Log associated reflector IDs
            if LOG_ASSOCIATED:
                current_assoc_ids = {
                    r.landmark.id
                    for r in reflectors
                    if r.associated and r.landmark.id != 0
                }
                new_ids = current_assoc_ids - seen_associated_ids
                if new_ids:
                    for rid in sorted(new_ids):
                        assoc_csv_writer.writerow([rid])
                    assoc_log_file.flush()
                    seen_associated_ids.update(new_ids)
                    print(f"[INFO] Logged {len(new_ids)} new associated ID(s): {sorted(new_ids)}")

            # Extract unassociated reflectors with non-zero world coordinates
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
          

            # Log newly detected unassociated reflectors
            if filteredReflectors and LOG_MISSING:
                print(f"Detected {len(filteredReflectors)} new unassociated reflectors:")

                LgvPosX = round(LgvPosX_symbol.read())
                LgvPosY = round(LgvPosY_symbol.read())

                for i, (wx, wy, assoc) in enumerate(filteredReflectors):
                    timestamp = datetime.now().isoformat(timespec='milliseconds')
                    world_x_mm = round(wx * 1000)
                    world_y_mm = round(wy * 1000)
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
            elif filteredReflectors:
                print(f"Detected {len(filteredReflectors)} new unassociated reflectors (logging disabled).")

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
        if log_file:
            log_file.close()
        if assoc_log_file:
            assoc_log_file.close()
        cleanup_old_logs()


if __name__ == "__main__":
    try:
        config_created, PLC_AMS_ID, PLC_IP, PORT, remoterun_enable, one_file, LOG_DURATION_HOURS, \
        VarReadInterval, log_dir, MAX_LOG_FOLDER_SIZE_MB, symbols_tc3, \
        LOG_MISSING, LOG_ASSOCIATED = load_configuration()

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