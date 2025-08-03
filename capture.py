import cv2
import socket

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cleint.connect(('127.0.0.1', 747))

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Cannot read from cap")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    cv2.imshow('Drone', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == "__main__":
    main()