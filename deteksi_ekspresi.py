import cv2
from deepface import DeepFace

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analisis ekspresi wajah
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in result:  # Bisa lebih dari 1 wajah
            # Ambil bounding box
            x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]

            # Ambil emosi dominan
            emotion = face["dominant_emotion"]

            # Gambar kotak di wajah
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0),
                          2)

            # Tampilkan teks di atas kotak
            cv2.putText(frame,
                        emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

    except Exception as e:
        print("Wajah tidak terdeteksi")

    # Tampilkan video dengan hasil deteksi
    cv2.imshow("Pendeteksi Ekspresi Wajah", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera
cap.release()
cv2.destroyAllWindows()
