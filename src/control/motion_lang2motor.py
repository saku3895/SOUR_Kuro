def parse_motion_commands(input_data):
    """
    動作言語をモーター制御値に変換する
    
    Args:
        input_data (tuple): ('20', ['*command1#', '*command2#', ...]) のような形式のタプル
        
    Returns:
        list: [[(motor_id, value), ...], ...] の各コマンドの結果リスト
    """
    
    # キーとモーター情報のマッピング
    motor_mapping = {
        'a': {'motor_id': 2, 'range': (850, 100), 'input_range': (0, 9)},
        'd': {'motor_id': 3, 'range': (0, 550), 'input_range': (3, 8)},
        'e': {'motor_id': 4, 'range': (500, 850), 'input_range': (0, 5)},
        'q': {'motor_id': 5, 'range': (850, 450), 'input_range': (5, 9)},
        'i': {'motor_id': 6, 'range': (100, 500), 'input_range': (5, 9)},
        'm': {'motor_id': 7, 'range': (500, 130), 'input_range': (0, 5)},
        'n': {'motor_id': 8, 'range': (150, 900), 'input_range': (0, 9)},
        'f': {'motor_id': 9, 'range': (100, 900), 'input_range': (0, 9)},
        'r': {'motor_id': 10, 'range': (850, 90), 'input_range': (0, 9)},
        'h': {'motor_id': 11, 'range': (200, 600), 'input_range': (2, 7)},
        'l': {'motor_id': 12, 'range': (850, 400), 'input_range': (2, 7)},
        'j': {'motor_id': 13, 'range': (850, 100), 'input_range': (0, 9)},
        'p': {'motor_id': 14, 'range': (450, 850), 'input_range': (2, 7)},
        't': {'motor_id': 15, 'range': (650, 150), 'input_range': (2, 7)}
    }
    
    def calculate_motor_value(key, input_value):
        """入力値からモーター値を計算"""
        if key not in motor_mapping:
            return None
            
        motor_info = motor_mapping[key]
        min_range, max_range = motor_info['range']
        min_input, max_input = motor_info['input_range']
        
        # 入力値が範囲外の場合はクランプ
        input_value = max(min_input, min(max_input, input_value))
        
        # 等分割による直接計算
        step_size = (max_range - min_range) / (max_input - min_input)
        motor_value = (input_value - min_input) * step_size + min_range
        
        return int(motor_value)
    
    results = []
    
    # タプルから番号とコマンドリストを取得
    if not isinstance(input_data, tuple) or len(input_data) != 2:
        return results
    
    sequence_number, command_list = input_data
    
    # コマンドリストの各要素を処理
    for command_string in command_list:
        # '*'で始まって'#'で終わる文字列から有効部分を抽出
        if command_string.startswith('*') and command_string.endswith('#'):
            # '*'と'#'を除去
            clean_command = command_string[1:-1]
            
            # 1つのコマンドの結果を格納するリスト
            command_results = []
            
            # '#'で区切られたセクションを処理
            sections = clean_command.split('#')
            
            for section in sections:
                if not section:
                    continue
                    
                i = 0
                while i < len(section):
                    if section[i].isalpha():
                        key = section[i]
                        i += 1
                        
                        # 数値部分を抽出
                        num_str = ""
                        while i < len(section) and section[i].isdigit():
                            num_str += section[i]
                            i += 1
                        
                        if num_str:
                            input_value = int(num_str)
                            motor_value = calculate_motor_value(key, input_value)
                            
                            if motor_value is not None:
                                motor_id = motor_mapping[key]['motor_id']
                                command_results.append((motor_id, motor_value))
                    else:
                        i += 1
            
            # 各コマンドの結果をメインのresultsリストに追加
            results.append(command_results)
    
    return results


# 使用例
if __name__ == "__main__":
    # テスト用のデータ
    test_data = ('20', ['*b5c0d5e7f2g2i6j2k2l3m4n3o0q5r3s1t0#', '*a6b4c5d7e0f5g6i0j5k3l5m1n6o5p7q1r5s4t7#', '*t8#'])

    print(f"入力データ: {test_data}")
    results = parse_motion_commands(test_data)

    for i, command_result in enumerate(results):
        print(f"コマンド {i+1}:")
        for motor_id, value in command_result:
            print(f"  モーター {motor_id}: {value}")
        print()
