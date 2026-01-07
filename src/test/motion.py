import copy
import random
class Motion():

    FLUCTUATION_TYPE_NONE = 0
    FLUCTUATION_TYPE_UNIFORM = 1

    def __init__(self, position, time, interval, loop_time=1, fluc_type=FLUCTUATION_TYPE_NONE, fluc_amp_motion=0.0, fluc_amp_time=0.0, fluc_amp_interval=0.0):
        assert len(position) == len(time) and len(time) == len(interval)

        self.position = position
        self.time = time
        self.interval = interval
        self.current_index = 0
        self.loop_count = 0
        self.loop_time = loop_time

        self.fluc_type = fluc_type
        self.fluc_amp_motion = fluc_amp_motion
        self.fluc_amp_time = fluc_amp_time
        self.fluc_amp_interval = fluc_amp_interval

        #for randomizer, prepare 2 dimensional array
        self.randomizer = [[None for j in range(len(self.position[0]))] for i in range(len(self.position))]

        self.motion_finished = False


    def get_next_motion(self):
        
        if self.motion_finished:
            return None, None, None

        next_index = self.current_index

        self.current_index = (self.current_index + 1) % len(self.position)
        
        if next_index == len(self.position)-1:
            self.loop_count += 1
            if self.loop_time > 0 and self.loop_count >= self.loop_time:#loop_time == 0 is infinite loop
                self.motion_finished = True

        next_position = copy.deepcopy(self.position[next_index])
        next_time = copy.deepcopy(self.time[next_index])
        next_interval = copy.deepcopy(self.interval[next_index])

        #set random value
        for i in range(len(next_position)):

            if self.randomizer[next_index][i] is not None:
                min_pos, max_pos = self.randomizer[next_index][i]
                next_position[i] = random.uniform(min_pos, max_pos)

        #set fluctuation
        for i in range(len(next_position)):

            if self.fluc_type == Motion.FLUCTUATION_TYPE_UNIFORM:
                next_position[i] += random.uniform(-self.fluc_amp_motion, self.fluc_amp_motion)
                next_time[i] += random.uniform(-self.fluc_amp_time, self.fluc_amp_time)
                next_interval += random.uniform(-self.fluc_amp_interval, self.fluc_amp_interval)

        return next_position, next_time, next_interval

    def get_motion_length(self):
        return len(self.position)

    def is_motion_finished(self):
        return self.motion_finished

    def reset(self):
        self.current_index = 0
        self.loop_count = 0
        self.motion_finished = False

    def set_randomizer(self, index_i, index_j, min_pos, max_pos):
        self.randomizer[index_i][index_j] = (min_pos, max_pos)