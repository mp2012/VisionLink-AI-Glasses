import cv2

# 遍历 0~5，找可用摄像头
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 强制用Windows兼容后端
    if cap.isOpened():
        print(f"✅ 找到摄像头，索引={i}")
        break
    cap.release()
else:
    print("❌ 没找到可用摄像头")
    exit()

# 先设一个肯定支持的分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 读不到帧")
        break

    cv2.imshow("CAM_TEST", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()
