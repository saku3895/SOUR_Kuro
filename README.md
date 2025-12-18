# SOUR
<p align="center">
    <img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/b431c4bf-9913-4f6d-8efd-fa7d2220e88e" />
</p>
Simple Operation for Ubiquitous Robotics (SOUR). Simple and easy to use middleware library for robotics

## Introduction
SOURはセットアップいらずで軽快に使えるロボット用ミドルウェアです．Pythonのみでコーディングでき，各種モジュールの再利用も可能です．
SOURではArduino等と同様に，基本的には定期的に実行される処理を記述します．また処理に応じてそれぞれの定期実行間隔を設定でき，
それらが並列に実行されます．そのため異なる種類のセンサ，ロボットのアクチュエータ制御，ロボットの意思決定などを別々に記述することができ，
管理が容易です．

SOURではRobotCoreに定期実行処理を記述したPeriodicModuleサブクラスを登録することで実行可能となります．
これらはマルチプロセスとして並列実行され，設定された定期実行間隔に応じて処理されます．
その仕組みにより厳密なリアルタイム性は保証されませんが，実行間隔を小さくすれば十分様々な用途に利用可能です．
PeriodicModuleのexecute_periodic_taskがArduinoのloopに相当します．

PeriodicModuleインスタンス間のデータやり取りは共有メモリdata_dictを通じて行われます．
<p align="center">
    <img width="400" height="252" alt="Image" src="https://github.com/user-attachments/assets/87fff997-5c1a-452b-8880-27a7bc7357f0" />
</p>

## Properties
RobotPropertiesインスタンスには下記情報を設定します．
- num_servos: サーボモータの個数
- servo_ids: サーボモータのID番号
- servo_min_positions: 最小角度のPWMパルス幅，または設定値
- servo_max_positions: 最小角度のPWMパルス幅，または設定値
- servo_initial_positions: サーボ角度の初期位置
- servo_shifts: サーボの初期位置のずれを是正するための移動量
- servo_easing_function: サーボ動作のイージング関数
- servo_controller_port: サーボ制御ボードとの通信ポート
- camera_port: カメラとの通信ポート（AIカメラを想定）

## Main Modules
#### PeriodicModule
- 定期実行されるモジュール，引数interval_msで指定した実行間隔(ミリ秒)でexecute_periodic_taskが実行される．このクラスを継承したサブクラスで，メソッドをオーバーライドして使う．
- terminate_allを呼び出すとすべてのPeriodicModuleを終了する．（呼び出しもとのPeriodicModule以外もすべて終了）
- RobotCoreインスタンスにregister_moduleで登録して利用する

#### PMServoControl
- サーボ制御用のPeriodicModule
- 共有メモリのdata_dictにservo_target_positions, servo_operation_timesとして数値のlistを書き込んで,かつservo_params_update = Trueとすると，servo_target_positionsで指定の角度に，servo_operation_timesで指定された時間(ミリ秒)で到達するようにサーボを駆動する．リストの数値は，RobotPropertiesのservo_idsに登録したID順に対応する．すべてのサーボの動作が完了すると，data_dictのservo_ready = Trueとなる．
- 実際のサーボ駆動部分は利用するサーボドライバなどに合わせてこのクラスを継承したサブクラスに記述する．サブクラスではなくこのクラスを直接利用すると，想定されるサーボの角度情報をコンソールに出力するため，デバッグに利用できる．

#### PMServoControlHiwonderSerialBusServoController
- Hiwonder Serial Bus Servo Controllerに対応したPMServoControlのサブクラス

#### PMUnitV2
- M5Stack UnitV2 AIカメラとの通信用PeriodicModule
- RobotPropertiesのcamera_portをUnitV2の接続ポートとして指定する
- read_dataでUnitV2から結果を受信する．現状ではその結果をprintしているのみ

## Usage
1. RobotPropertiesインスタンスの各種パラメータを適切に設定し，それを引数としてRobotCoreインスタンスを作成する
1. RobotCoreインスタンスに必要なセンサやアクチュエータ用のPeriodicModuleインスタンスを登録する
1. RobotCoreインスタンスにロボットの行動制御用PeriodicModuleインスタンスを登録する
1. RobotCoreインスタンスのrun()を実行


## Environment
- Python 3.x~
- smbus2
