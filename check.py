import cv2
import face_recognition
import pyodbc
import json
import numpy as np
import time

# --- CONFIGURATION ---
SERVER_NAME = 'localhost'
DB_NAME = 'attendance_ai'
TOLERANCE = 0.6 
SNAPSHOT_INTERVAL = 5    # Seconds between photos
TOTAL_SNAPSHOTS = 6      # Total photos to take
PASSING_THRESHOLD = 3    # Min photos to be considered "Present"

# --- DATABASE CONNECTION ---
def get_db_connection():
    return pyodbc.connect(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={SERVER_NAME};'
        f'DATABASE={DB_NAME};'
        'Trusted_Connection=yes;'
    )

def log_to_db(uid, name, subject, period, date_str, status, notes=""):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # # Check if already logged to avoid duplicates
        # cursor.execute("SELECT log_id FROM attendance_log WHERE student_uid = ? AND subject_name = ? AND log_date = ?", 
        #                (str(uid), subject, date_str))
        # if cursor.fetchone():
        #     # (Optional) Update existing record if needed, or just skip
        #     # For now, we skip to avoid double entries
        #     return

        # clean_uid = ''.join(filter(str.isdigit, str(uid))) 
        final_status = f"{status} ({notes})" if notes else status

        sql = """
            INSERT INTO attendance_log 
            (student_uid, student_name, subject_name, period_number, log_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        values = (clean_uid, name, subject, period, date_str, final_status)
        
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
        print(f" [DB] SAVED: {name} -> {final_status}")
        
    except Exception as e:
        print(f" [DB] ERROR: {e}")

def get_all_students():
    """ Load all students and return lists """
    print("\n--- LOADING CLASS DATABASE ---")
    known_encodings = []
    known_names = []
    known_uids = []

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT uid, name, encoding FROM face_data")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            known_uids.append(row[0])
            known_names.append(row[1])
            known_encodings.append(np.array(json.loads(row[2])))
            
        print(f"Loaded {len(known_names)} students.")
        return known_uids, known_names, known_encodings
    except Exception as e:
        print(f"Database Error: {e}")
        return [], [], []

def run_class_session():
    # 1. SETUP
    print("--- AUTOMATED CLASS SESSION ---")
    subject = input("Enter Subject: ")
    period  = input("Enter Period: ")
    date_input = input("Enter Date (YYYY-MM-DD): ")
    
    # 2. LOAD ALL STUDENTS
    all_uids, all_names, all_encodings = get_all_students()
    
    # 3. INITIALIZE TRACKING DICTIONARY
    # Structure: { 'U101': [False, False, ...], 'U102': [False, False, ...] }
    attendance_sheet = {uid: [False]*TOTAL_SNAPSHOTS for uid in all_uids}
    
    # Map UID to Name for easy lookup later
    uid_to_name = {uid: name for uid, name in zip(all_uids, all_names)}

    # 4. START SESSION LOOP
    cam = cv2.VideoCapture(0)
    print(f"\nSTARTING SESSION: Taking {TOTAL_SNAPSHOTS} photos (1 every {SNAPSHOT_INTERVAL}s)...")
    
    for i in range(TOTAL_SNAPSHOTS):
        start_time = time.time()
        
        # Countdown Loop for visual feedback
        while True:
            ret, frame = cam.read()
            if not ret: break
            
            time_left = int((start_time + SNAPSHOT_INTERVAL) - time.time())
            
            if time_left <= 0:
                break # Time to snap!

            # UI
            cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), -1)
            cv2.putText(frame, f"Photo {i+1}/{TOTAL_SNAPSHOTS}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Snap in: {time_left}s", (350, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Class Scanner", frame)
            cv2.waitKey(1)

        # --- SNAPSHOT TAKEN ---
        print(f"Processing Snapshot {i+1}...")
        
        # 1. Detect all faces in frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 2. Identify who they are
        found_in_this_frame = [] # Keep track of who we saw in THIS specific photo
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(all_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(all_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                found_uid = all_uids[best_match_index]
                found_in_this_frame.append(found_uid)

        # 3. Update the Attendance Sheet
        for uid in all_uids:
            if uid in found_in_this_frame:
                attendance_sheet[uid][i] = True
                print(f"  -> Found: {uid_to_name[uid]}")
    
    cam.release()
    cv2.destroyAllWindows()

    # 5. CALCULATE FINAL RESULTS FOR EVERYONE
    print("\n" + "="*40)
    print(" FINAL CLASS REPORT")
    print("="*40)

    for uid in all_uids:
        record = attendance_sheet[uid] # e.g. [True, True, True, False...]
        present_count = sum(record)
        name = uid_to_name[uid]
        
        status = "ABSENT"
        notes = ""

        # LOGIC ENGINE
        if present_count < PASSING_THRESHOLD:
            status = "ABSENT"
            if present_count > 0: notes = "Low Attendance"
        else:
            status = "PRESENT"
            
            # Check Patterns
            start_missing = (not record[0] and not record[1])
            end_missing   = (not record[-1] and not record[-2])

            if start_missing:
                status = "LATE ENTRY"
                notes = "Missed start"
            elif end_missing:
                status = "EARLY EXIT"
                notes = "Left early"

        print(f" {name}: {status} {record}")
        
        # 6. LOG TO DATABASE
        log_to_db(uid, name, subject, period, date_input, status, notes)

if __name__ == "__main__":
    run_class_session()
