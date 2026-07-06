import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Inisialisasi MediaPipe Hand Landmarker menggunakan Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

# Konfigurasi kamera yang tersedia untuk dicoba (memprioritaskan MSMF dan Default dibanding DSHOW)
camera_configs = [
    (0, cv2.CAP_MSMF),   # MSMF index 0 (Sangat stabil & bersih untuk kamera RGB Windows)
    (0, None),           # Default index 0
    (1, cv2.CAP_MSMF),   # MSMF index 1 (Kamera RGB eksternal/kedua)
    (1, None),           # Default index 1
    (0, cv2.CAP_DSHOW),  # DirectShow index 0 (Fallback jika MSMF bermasalah)
    (1, cv2.CAP_DSHOW),  # DirectShow index 1 (Fallback jika MSMF bermasalah)
    (2, cv2.CAP_MSMF),   # MSMF index 2
    (2, None),           # Default index 2
]

config_idx = 0

def open_camera(idx):
    index, backend = camera_configs[idx]
    backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else ("MSMF" if backend == cv2.CAP_MSMF else "Default")
    print(f"Mencoba membuka kamera index {index} dengan backend {backend_name}...")
    
    if backend is not None:
        cap = cv2.VideoCapture(index, backend)
    else:
        cap = cv2.VideoCapture(index)
        
    if cap.isOpened():
        # Lakukan pemanasan kamera dengan mencoba membaca frame beberapa kali
        for _ in range(10):
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                print(f"Kamera Berhasil Dibuka! Index: {index}, Backend: {backend_name}")
                return cap
            time.sleep(0.1)
        cap.release()
    print(f"Gagal membuka/membaca frame dari kamera index {index} ({backend_name}).")
    return None

# Cari kamera pertama yang bisa dibuka
cap = None
for i in range(len(camera_configs)):
    cap = open_camera(i)
    if cap is not None:
        config_idx = i
        break

if cap is None:
    print("Error: Tidak ada kamera yang bisa dibuka. Pastikan webcam terhubung dan tidak digunakan aplikasi lain.")
    exit(1)

# Menggunakan konteks manager untuk memastikan resource Landmarker dibebaskan
with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("\n=== PROGRAM BERJALAN ===")
    print("- Tekan 'q' di jendela kamera untuk Keluar")
    print("- Tekan 'n' di jendela kamera untuk ganti Kamera/Backend (jika layar abu-abu/rusak/salah kamera)\n")
    
    current_blur_amount = 0.0
    TRANSITION_SPEED = 0.15

    while True:
        if cap is None or not cap.isOpened():
            print("Kamera terputus atau ditutup. Mencoba konfigurasi berikutnya...")
            cap = None
            for offset in range(1, len(camera_configs) + 1):
                next_idx = (config_idx + offset) % len(camera_configs)
                cap = open_camera(next_idx)
                if cap is not None:
                    config_idx = next_idx
                    break
            if cap is None:
                print("Error: Tidak menemukan kamera alternatif yang berfungsi.")
                break

        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = landmarker.detect(mp_image)

        blur_effect = False

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                index_tip = hand_landmarks[8].y
                index_pip = hand_landmarks[6].y
                
                middle_tip = hand_landmarks[12].y
                middle_pip = hand_landmarks[10].y
                
                ring_tip = hand_landmarks[16].y
                ring_pip = hand_landmarks[14].y
                
                pinky_tip = hand_landmarks[20].y
                pinky_pip = hand_landmarks[18].y

                # Logika Pose 2 Jari: Telunjuk & Tengah naik, Manis & Kelingking turun
                if (index_tip < index_pip and 
                    middle_tip < middle_pip and 
                    ring_tip > ring_pip and 
                    pinky_tip > pinky_pip):
                    blur_effect = True

        # Update tingkat blur secara halus (transisi)
        if blur_effect:
            current_blur_amount = min(1.0, current_blur_amount + TRANSITION_SPEED)
        else:
            current_blur_amount = max(0.0, current_blur_amount - TRANSITION_SPEED)

        # Terapkan efek blur jika transisi sedang aktif
        if current_blur_amount > 0.0:
            # Petakan tingkat transisi (0.0 ke 1.0) menjadi ukuran kernel ganjil (1 ke 99)
            ksize = int(1 + (98 * current_blur_amount))
            if ksize % 2 == 0:
                ksize += 1
            if ksize > 1:
                frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        cv2.putText(frame, "Tekan 'q': Keluar | Tekan 'n': Ganti Kamera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        index, backend = camera_configs[config_idx]
        backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else ("MSMF" if backend == cv2.CAP_MSMF else "Default")
        cv2.putText(frame, f"Kamera Aktif: Index {index} ({backend_name})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Camera Filter', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            print("\nMengganti kamera atas permintaan pengguna...")
            cv2.destroyWindow('Camera Filter')
            cap.release()
            cap = None
            for offset in range(1, len(camera_configs) + 1):
                next_idx = (config_idx + offset) % len(camera_configs)
                next_cap = open_camera(next_idx)
                if next_cap is not None:
                    cap = next_cap
                    config_idx = next_idx
                    break
            if cap is None:
                print("Error: Tidak dapat menemukan kamera lain yang berfungsi.")
                break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()

