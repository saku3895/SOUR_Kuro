import serial
import time

class LobotServoController:
    """
    LSCシリーズサーボモーターコントローラ用のPythonクラス
    """
    
    # フレームヘッダーと命令コード
    FRAME_HEADER = 0x55
    CMD_SERVO_MOVE = 0x03
    CMD_ACTION_GROUP_RUN = 0x06
    CMD_ACTION_GROUP_STOP = 0x07
    CMD_ACTION_GROUP_SPEED = 0x0B
    CMD_GET_BATTERY_VOLTAGE = 0x0F
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=0.1):
        """
        LobotServoControllerのインスタンスを初期化
        
        Args:
            port: シリアルポート名 (デフォルトはラズパイ用の設定)
            baudrate: ボーレート
            timeout: シリアル通信のタイムアウト (秒)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False
        self.battery_volt = 0
        
    def connect(self):
        """シリアルポートに接続"""
        try:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False
            return False
            
    def disconnect(self):
        """シリアルポートとの接続を閉じる"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print("Disconnected")
            
    def _check_connection(self):
        """接続状態をチェック"""
        if not self.is_connected or not self.ser or not self.ser.is_open:
            raise ConnectionError("Not connected to serial port. Call connect() first.")
            
    @staticmethod
    def _get_low_byte(value):
        """16ビット値の下位8ビットを取得"""
        return value & 0xFF
        
    @staticmethod
    def _get_high_byte(value):
        """16ビット値の上位8ビットを取得"""
        return (value >> 8) & 0xFF

    def moveServo(self, servo_id, position, move_time):
        """
        単一のサーボモーターを制御
        
        Args:
            servo_id: サーボID (0-31)
            position: 目標位置 (0-1000)
            move_time: 移動時間 (ms)
        """
        self._check_connection()
        
        if servo_id > 31 or move_time <= 0:
            print("Invalid servo ID or move time")
            return
            
        tx_buf = bytearray([
            self.FRAME_HEADER,
            self.FRAME_HEADER,
            8,  # データ長 = 制御するサーボ数*3+5 = 1*3+5 = 8
            self.CMD_SERVO_MOVE,
            1,  # 制御するサーボの数
            self._get_low_byte(move_time),
            self._get_high_byte(move_time),
            servo_id,
            self._get_low_byte(position),
            self._get_high_byte(position)
        ])
        
        self.ser.write(tx_buf)
        print(f"Moving servo ID {servo_id} to position {position} in {move_time}ms")
        
    def moveServos(self, servos, move_time):
        """
        複数のサーボモーターを制御
        
        Args:
            servos: (servo_id, position)タプルのリスト
            move_time: 移動時間 (ms)
        """
        self._check_connection()
        
        num_servos = len(servos)
        if num_servos < 1 or num_servos > 32 or move_time <= 0:
            print("Invalid number of servos or move time")
            return
            
        # フレームヘッダーなど基本情報を設定
        tx_buf = bytearray([
            self.FRAME_HEADER,
            self.FRAME_HEADER,
            num_servos * 3 + 5,  # データ長 = 制御するサーボ数*3+5
            self.CMD_SERVO_MOVE,
            num_servos,
            self._get_low_byte(move_time),
            self._get_high_byte(move_time)
        ])
        
        # サーボ情報を追加
        for servo_id, position in servos:
            tx_buf.extend([
                servo_id,
                self._get_low_byte(position),
                self._get_high_byte(position)
            ])
            
        self.ser.write(tx_buf)

    def getBatteryVoltage(self):
        """バッテリー電圧を取得"""
        self._check_connection()
        
        tx_buf = bytearray([
            self.FRAME_HEADER,
            self.FRAME_HEADER,
            2,  # データ長
            self.CMD_GET_BATTERY_VOLTAGE
        ])
        
        self.ser.write(tx_buf)
        
        # 応答を待機
        time.sleep(0.1)
        
        if self.ser.in_waiting >= 6:
            rx_buf = self.ser.read(6)
            if rx_buf[0] == self.FRAME_HEADER and rx_buf[1] == self.FRAME_HEADER and rx_buf[3] == self.CMD_GET_BATTERY_VOLTAGE:
                self.battery_volt = (rx_buf[5] << 8) | rx_buf[4]
                print(f"Battery voltage: {self.battery_volt/1000.0:.2f}V")
                return self.battery_volt/1000.0
        
        print("Failed to get battery voltage")
        return None


# 使用例
if __name__ == "__main__":
    myse = LobotServoController(port='/dev/ttyAMA0')

    if myse.connect():
        try:
            # 最初にバッテリー電圧を取得
            voltage = myse.getBatteryVoltage()
            if voltage is not None:
                print(f"現在のバッテリー電圧: {voltage:.2f}V")

            # 2つのサーボを同時に制御
            # [(サーボID, 位置), (サーボID, 位置)]の形式でリストを渡す
            myse.moveServos([(2, 200), (3, 500)], 1000)  # サーボ3と4を同時に移動 1s
            time.sleep(2.0)
            myse.moveServos([(2, 600), (3, 300)], 1000)  # 逆方向に移動 1s
            time.sleep(2.0)

        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        finally:
            myse.disconnect()
