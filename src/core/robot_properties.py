class RobotProperties:
    def __init__(self):
        self.num_servos = 0
        self.servo_ids = []
        self.servo_min_positions = []
        self.servo_max_positions = []
        self.servo_shifts = []

        self.servo_easing_function = ServoEasingFunctions.LINEAR
        self.servo_initial_positions = []

        self.servo_controller_port = '/dev/ttyAMA1'


class ServoEasingFunctions:
    #define constant
    LINEAR = 0
    EASE_IN_OUT_CUBIC = 1

    def __init__(self):
        pass

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def ease_in_out_cubic(x):
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2