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
#### PMServoControl
- サーボ制御用のPeriodicModule

## Environment
- Python 3.x~
- smbus2
