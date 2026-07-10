import cv2
import depthai as dai

pipeline = dai.Pipeline()

# 1. カラーカメラノードの作成
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# 💡 ここが超重要！
# 横長（1080p）の動画出力を使う場合は、以下の解像度設定が「必須」です
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 2. カメラ本来のフルHD横長映像（video）を出口に繋ぐ
camRgb.video.link(xoutVideo.input)

# 3. カメラの起動
with dai.Device(pipeline) as device:
    # フルHD映像はデータ量が大きいので、maxSizeを少し余裕を持たせて4に、blockingをFalseにします
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    print("OAK-D S2 フルHD（横長）モード起動成功！")
    while True:
        inVideo = videoQueue.get()
        
        # 画面にフルHDのままだと大きすぎる場合があるので、少し縮小して表示（不要ならこの行を消してinVideo.getCvFrame()を直接imshowしてください）
        frame = inVideo.getCvFrame()
        resized_frame = cv2.resize(frame, (960, 540)) # 画面に収まりやすいサイズに半分に縮小
        
        cv2.imshow("OAK-D S2 Full HD Video", resized_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()