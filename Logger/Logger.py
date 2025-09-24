import pyads
import ctypes
import time
import csv
import os
from datetime import datetime
import glob
import re

#Log file configuration
log_dir = r"D:\Logs\MissingReflectors"
MAX_LOG_FOLDER_SIZE_MB = 20
os.makedirs(log_dir, exist_ok=True)

#Connection to PLC
PLC_AMS_ID = '192.168.11.2.1.1'
PLC_IP = '192.168.11.2'
PORT = 851


NumReflectors = 50
VarReadInterval = 1000  # ms
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
    


def main():
    next_time = time.perf_counter()
    current_hour = datetime.now().hour
    log_file = open(get_log_path(lgv_num), mode='w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Lgv","Timestamp","WorldX", "WorldY", "LgvX", "LgvY"])
    # -----------------------------
    # Connect to PLC
    # -----------------------------
    plc = pyads.Connection(PLC_AMS_ID, PORT, PLC_IP)
    plc.open()

    # Get symbols for efficient access
    avoid_reflector_symbol = plc.get_symbol("CustomPlcAttribute.AvoidReflectorCheck_sp")
    quality_symbol = plc.get_symbol("Sys_ExternalLocalization.extPoseInfo.quality")
    Aut_Run_symbol = plc.get_symbol("LibraryInterfaces.LGV.Status.Aut_Run")

    #Get symbols for LGV coordinates
    LgvPosX_symbol = plc.get_symbol("LibraryInterfaces.LGV.Guid.Info.Pos.X")
    LgvPosY_symbol = plc.get_symbol("LibraryInterfaces.LGV.Guid.Info.Pos.Y")


    previousReflectors = []

    try:
        while True:
            now = datetime.now()
            if now.hour != current_hour:
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
                
                if not avoid_reflector and Aut_Run and quality > 0.8:
                    print("Reading reflectors...")
                    raw_data_list = plc.read_by_name(
                        "Sys_ExternalLocalization.extReflectorSet[1].reflectors",
                        ctypes.c_ubyte * (StructSize * NumReflectors)  # Use c_ubyte for bytes 0..255
                    )

                    LgvPosX = round(LgvPosX_symbol.read())
                    LgvPosY = round(LgvPosY_symbol.read())

                else:
                    if avoid_reflector:print("AvoidReflectorCheck_sp... skipping")
                    elif not Aut_Run: print("LGV not in Auto... skipping")
                    else: print(f'Low quality({quality:.3f})... skipping')
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

if __name__ == "__main__":
    lgv_num = extract_lgv_number()
    main()


# """NEXT STEPS:
#         - In post processing we could:
#             - Cluster nearby reflectors (DBSCAN?)
#             - Define a score with different LGVs to define which reflectors are truly missing
#             - Mark in the dxf which are missing, along with their mapping coordinates
#                   - Taking into account the relative position of the LGV to the reflector and averaging
#                   - Taking into account max absolute distance from LGV to reflector 
#                   - Use DBSCAN to define clusters. Average the coordinates to map
#                   - Use score from different time, lgv, angle? to determine confidence and color
#                   - Compare with reflectors in layout
#                       - Mark if they're missing or just moved
#                       * import sqlite3
#                       * conn = sqlite3.connect(r'')
#                       * cur = conn.cursor()
#                       * cur.execute("SELECT ID, X, Y FROM Reflectors")
#                       * rows = cur.fetchall()
#                       * for row in rows: print(row)
#                       * conn.close()
#                   - For the average to map, use an average of the average of each lgv
#                       * Calculate lgv mean - consensus (mean of means) per each reflector
#                       * Determine a threshold to flag big biases (std or vector (x+y) > refl diameter/2)
#                       * In flagged per lgv, check if it's consistent: similar bias, most cases
#                       * Flag that lgv for calibration.
#                           > If similar bias, apply correction and compute means again
#                           > If different bias, reduce weight and compute means again
#                       * Minimum amount of reflectors (5?) needed to calculate bias
#                       * Extra: calculate angle between lgv and reflector and add pole radius to pos
#           - Apply similar analysis with different logs (all reflectors) to determine lgvs in need
#               of calibration
# """