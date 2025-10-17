import pyads
import ctypes
import time
import csv
import os
from datetime import datetime
import glob
import re
import xml.etree.ElementTree as ET


NumReflectors = 50
StructSize = 88  # bytes per ReflectorInfo

# -----------------------------
# Define ctypes structs
# -----------------------------
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

def get_log_path(lgvNum):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"LGV{lgvNum}_MissingReflector_log_{timestamp}.csv"
    return os.path.join(log_dir, filename)

def get_log_folder_size_mb():
    total_size = 0
    for f in glob.glob(os.path.join(log_dir, "*.csv")):
        total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)

def cleanup_old_logs():
    files = sorted(
        glob.glob(os.path.join(log_dir, "*.csv")),
        key=os.path.getmtime
    )
    while get_log_folder_size_mb() > MAX_LOG_FOLDER_SIZE_MB and files:
        oldest = files.pop(0)
        print(f"Deleting old log file: {oldest}")
        os.remove(oldest)

def extract_lgv_number():
    with open(r'D:\Config\LGV.XML', 'r') as file:
        text = file.read().strip()
    
    match = re.search(r'LGV(\d{1,2})', text)
    if match:
        extracted_number = int(match.group(1))
        print(f"Extracted lgv number: {extracted_number}")
        return extracted_number
    else:
        raise ValueError("No lgv number found.")
    

def generate_configfile():
    root = ET.Element("Configuration")
    comm = ET.SubElement(root, "Communication")
    ET.SubElement(comm, "AmsNetId").text = '192.168.11.2.1.1'
    ET.SubElement(comm, "IpAddress").text = '192.168.11.2'
    ET.SubElement(comm, "Port").text = '851'

    log = ET.SubElement(root, "Log_config")
    ET.SubElement(log, "OneFile").text = 'True'
    ET.SubElement(log, "LogDurationHours").text = '8'
    ET.SubElement(log, "ReadIntervalMs").text = '1000'
    ET.SubElement(log, "LogFolder").text = r'D:\Logs\MissingReflectors'
    ET.SubElement(log, "MaxLogFolderSizeMB").text = '20'

    
    symbols = ET.SubElement(root, "Symbols")
    avoid_reflector = ET.SubElement(symbols, "AvoidReflectorCheck")
    ET.SubElement(avoid_reflector, "Symbol").text = "CustomPlcAttribute.AvoidReflectorCheck_sp"
    ET.SubElement(avoid_reflector, "Bypass").text = 'False'

    quality = ET.SubElement(symbols, "Quality")
    ET.SubElement(quality, "Symbol").text = "Sys_ExternalLocalization.extPoseInfo.quality"
    ET.SubElement(quality, "Bypass").text = 'False'

    aut_run = ET.SubElement(symbols, "Aut_Run")
    ET.SubElement(aut_run, "Symbol").text = "LibraryInterfaces.LGV.Status.Aut_Run"
    ET.SubElement(aut_run, "Bypass").text = 'False'

    man_run = ET.SubElement(symbols, "Man_Run")
    ET.SubElement(man_run, "Symbol").text = "LibraryInterfaces.LGV.Status.Man_Run"
    ET.SubElement(man_run, "Bypass").text = 'False'

    isNotMoving = ET.SubElement(symbols, "IsNotMoving")
    ET.SubElement(isNotMoving, "Symbol").text = "LibraryInterfaces.LGV.Status.IsNotMoving"
    ET.SubElement(isNotMoving, "Bypass").text = 'False'

    lgvPosX = ET.SubElement(symbols, "LgvPosX")
    ET.SubElement(lgvPosX, "Symbol").text = "LibraryInterfaces.LGV.Guid.Info.Pos.X"

    lgvPosY = ET.SubElement(symbols, "LgvPosY")
    ET.SubElement(lgvPosY, "Symbol").text = "LibraryInterfaces.LGV.Guid.Info.Pos.Y"

    tree = ET.ElementTree(root)
    tree.write(r'D:\Config\ReflectorFinderConfig.xml', encoding='utf-8', xml_declaration=True)

def load_configuration():
    config_path = r'D:\Config\ReflectorFinderConfig.xml'
    if not os.path.exists(config_path):
        print("[WARNING] Configuration file not found. Generating default config.")
        generate_configfile()

    tree = ET.parse(config_path)
    root = tree.getroot()

    plc_ams_id = root.find("Communication/AmsNetId").text
    plc_ip = root.find("Communication/IpAddress").text
    port = int(root.find("Communication/Port").text)

    one_file = root.find("Log_config/OneFile").text.lower() == 'true'
    log_duration = int(root.find("Log_config/LogDurationHours").text)
    read_interval = int(root.find("Log_config/ReadIntervalMs").text)
    log_folder = root.find("Log_config/LogFolder").text
    max_log_size = int(root.find("Log_config/MaxLogFolderSizeMB").text)

    symbols = {}
    for symbol in root.find("Symbols"):
        name = symbol.tag
        symb_text = symbol.find("Symbol").text
        bypass_elem = symbol.find("Bypass")
        bypass = bypass_elem is not None and bypass_elem.text.lower() == 'true'
        symbols[name] = (symb_text, bypass)

    return plc_ams_id, plc_ip, port, one_file, log_duration, read_interval, log_folder, max_log_size, symbols

    


def main():
    if one_file:print(f"[INFO] One file mode. Logging for {LOG_DURATION_HOURS} hours.")

    next_time = time.perf_counter()
    current_hour = datetime.now().hour
    log_file = open(get_log_path(lgv_num), mode='w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Lgv","Timestamp","WorldX", "WorldY", "LgvX", "LgvY"])

    #For one file mode (8 hours and then stop)
    start_time = time.perf_counter()
    end_time = start_time + (LOG_DURATION_HOURS * 3600) if one_file else None
    # -----------------------------
    # Connect to PLC
    # -----------------------------
    plc = pyads.Connection(PLC_AMS_ID, PORT, PLC_IP)
    plc.open()

    # Get symbols
    avoid_reflector_symbol = plc.get_symbol(symbols["AvoidReflectorCheck"][0])
    quality_symbol = plc.get_symbol(symbols["Quality"][0])
    Aut_Run_symbol = plc.get_symbol(symbols["Aut_Run"][0])
    Man_Run_symbol = plc.get_symbol(symbols["Man_Run"][0])
    isNotMoving_symbol = plc.get_symbol(symbols["IsNotMoving"][0])
    LgvPosX_symbol = plc.get_symbol(symbols["LgvPosX"][0])
    LgvPosY_symbol = plc.get_symbol(symbols["LgvPosY"][0])

    #Get bypass flags
    avoid_reflector_bypass = symbols["AvoidReflectorCheck"][1]
    quality_bypass = symbols["Quality"][1]
    Aut_Run_bypass = symbols["Aut_Run"][1]
    Man_Run_bypass = symbols["Man_Run"][1]
    isNotMoving_bypass = symbols["IsNotMoving"][1]

    previousReflectors = []

    try:
        #initializing variables
        avoid_reflector = False
        quality = 0.0
        Aut_Run = False
        Man_Run = False
        isNotMoving = True

        while True:
            # Exit after 8h for one file mode
            if one_file and end_time is not None and time.perf_counter() >= end_time:
                print("Reached max log duration. Stopping.")
                break

            now = datetime.now()
            if now.hour != current_hour and not one_file:
                log_file.close()
                cleanup_old_logs()
                log_file = open(get_log_path(lgv_num), mode='w', newline='')
                csv_writer = csv.writer(log_file)
                csv_writer.writerow(["Lgv","Timestamp","WorldX", "WorldY", "LgvX", "LgvY"])
                current_hour = now.hour
                print(f"Started new log file at {now.strftime('%Y-%m-%d %H:%M:%S')}")

            # Read entire array of structs in one ADS request
            try:
                avoid_reflector = avoid_reflector_symbol.read()
                quality = quality_symbol.read()
                Aut_Run = Aut_Run_symbol.read()
                Man_Run = Man_Run_symbol.read()
                isNotMoving = isNotMoving_symbol.read()

                # Check bypasses
                avoidref_ok = avoid_reflector_bypass or not avoid_reflector
                quality_ok = quality_bypass or quality > 0.8
                run_ok = (Aut_Run_bypass and Man_Run_bypass) or Aut_Run or Man_Run
                notmoving_ok = isNotMoving_bypass or not isNotMoving
                
                if avoidref_ok and run_ok and notmoving_ok and quality_ok:
                    print("Reading reflectors...")
                    raw_data_list = plc.read_by_name(
                        "Sys_ExternalLocalization.extReflectorSet[1].reflectors",
                        ctypes.c_ubyte * (StructSize * NumReflectors)  # Use c_ubyte for bytes 0..255
                    )

                else:
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
                    continue  # skip this cycle and retry
            except Exception as unexpected:
                print(f"Unexpected error: {unexpected}")
                time.sleep(1)
                continue

            raw_data = bytes(raw_data_list)  # Convert list of c_ubyte to bytes

            # Convert raw bytes to array of ReflectorInfo structs
            ReflectorArray = ReflectorInfo * NumReflectors
            reflectors = ReflectorArray.from_buffer_copy(raw_data)

            newReflectors = [
                (r.worldX, r.worldY, bool(r.associated))
                for r in reflectors
                if r.worldX != 0.0 and r.associated == False
            ]

            # Drop entries that are exactly the same as in the previous loop
            filteredReflectors = []
            for ref in newReflectors:
                if ref not in previousReflectors:
                    filteredReflectors.append(ref)


            # Log results
            if filteredReflectors:
                print(f"Detected {len(filteredReflectors)} new unassociated reflectors:")

                LgvPosX = round(LgvPosX_symbol.read())
                LgvPosY = round(LgvPosY_symbol.read())
                
                for i, (wx, wy, assoc) in enumerate(filteredReflectors):
                    timestamp = datetime.now().isoformat(timespec='milliseconds')
                    csv_writer.writerow([
                        lgv_num,
                        timestamp,
                        f"{round(wx*1000)}",
                        f"{round(wy*1000)}",
                        LgvPosX,
                        LgvPosY
                    ])
                    print(f"Reflector {i+1}: worldX={round(wx*1000)}, worldY={round(wy*1000)}")
                
                log_file.flush()

            previousReflectors = newReflectors

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
    PLC_AMS_ID, PLC_IP, PORT, one_file, LOG_DURATION_HOURS, VarReadInterval, log_dir, MAX_LOG_FOLDER_SIZE_MB, symbols = load_configuration()
    
    os.makedirs(log_dir, exist_ok=True)
    
    
    lgv_num = extract_lgv_number()
    
    main()