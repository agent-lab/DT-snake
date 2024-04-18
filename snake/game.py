import errno
import os
import traceback
from enum import Enum, unique

from snake.base import Direc, Map, PointType, Pos, Snake
from snake.gui import GameWindow
import cv2
import time
import numpy as np
# Add solver names to globals()
from snake.solver import DQNSolver, GreedySolver, HamiltonSolver, WeakGreedySolver, WorstGreedySolver
from typing import Dict, Tuple, List
import pickle

@unique
class GameMode(Enum):
    NORMAL = 0  # AI with GUI
    BENCHMARK = 1  # Run benchmarks without GUI
    TRAIN_DQN = 2  # Train DQNSolver without GUI
    TRAIN_DQN_GUI = 3  # Train DQNSolver with GUI


class GameConf:
    def __init__(self):
        """Initialize a default configuration."""

        # Game mode
        self.mode = GameMode.NORMAL

        # Solver
        self.solver_name = "HamiltonSolver"  # Class name of the solver

        # output
        self.game_log_file = "data"

        # Size
        self.map_rows = 8
        self.map_cols = self.map_rows
        self.map_width = 160  # pixels
        self.map_height = self.map_width
        self.info_panel_width = 155  # pixels
        self.window_width = self.map_width + self.info_panel_width
        self.window_height = self.map_height
        self.grid_pad_ratio = 0.25

        # Switch
        self.show_grid_line = False
        self.show_info_panel = True

        # Delay
        self.interval_draw = 50  # ms
        self.interval_draw_max = 200  # ms

        # Color
        self.color_bg = "#000000"
        self.color_txt = "#F5F5F5"
        self.color_line = "#424242"
        self.color_wall = "#F5F5F5"
        self.color_food = "#FFF59D"
        self.color_head = "#F5F5F5"
        self.color_body = "#F5F5F5"

        # Initial snake
        self.init_direc = Direc.RIGHT
        self.init_bodies = [Pos(1, 4), Pos(1, 3), Pos(1, 2), Pos(1, 1)]
        self.init_types = [PointType.HEAD_R] + [PointType.BODY_HOR] * 3

        # Font
        self.font_info = ("Arial", 9)

        # Info
        self.info_str = (
            "<w/a/s/d>: snake direction\n"
            "<space>: pause/resume\n"
            "<r>: restart    <esc>: exit\n"
            "-----------------------------------\n"
            "status: %s\n"
            "episode: %d   step: %d\n"
            "length: %d/%d (" + str(self.map_rows) + "x" + str(self.map_cols) + ")\n"
            "-----------------------------------"
        )
        self.info_status = ["eating", "dead", "full"]


class Game:
    def __init__(self, conf):
        self._conf = conf
        self._map = Map(conf.map_rows + 2, conf.map_cols + 2)
        self._snake = Snake(
            self._map, conf.init_direc, conf.init_bodies, conf.init_types
        )
        self._pause = False
        self._solver = globals()[self._conf.solver_name](self._snake)
        self._episode = 1
        self._init_log_file()
        self._reward = 0
        self._game_log: Dict[int, Dict[str, List]] = {}
        self._game_log[self._episode] = {"reward":[], "action":[], "state":[], "RTG": []}

    @property
    def snake(self):
        return self._snake

    @property
    def episode(self):
        return self._episode

    def run(self):
        if self._conf.mode == GameMode.BENCHMARK:
            self._run_benchmarks()
        elif self._conf.mode == GameMode.TRAIN_DQN:
            self._run_dqn_train()
            self._plot_history()
        else:
            window = GameWindow(
                "Snake",
                self._conf,
                self._map,
                self,
                self._on_exit,
                (
                    ("<w>", lambda e: self._update_direc(Direc.UP)),
                    ("<a>", lambda e: self._update_direc(Direc.LEFT)),
                    ("<s>", lambda e: self._update_direc(Direc.DOWN)),
                    ("<d>", lambda e: self._update_direc(Direc.RIGHT)),
                    ("<r>", lambda e: self._reset()),
                    ("<space>", lambda e: self._toggle_pause()),
                ),
            )
            if self._conf.mode == GameMode.NORMAL:
                window.show(self._game_main_normal)
            elif self._conf.mode == GameMode.TRAIN_DQN_GUI:
                window.show(self._game_main_dqn_train)
                self._plot_history()

    def _run_benchmarks(self):
        steps_limit = 5000
        num_episodes = int(input("Please input the number of episodes: "))

        print(f"\nMap size: {self._conf.map_rows}x{self._conf.map_cols}")
        print(f"Solver: {self._conf.solver_name[:-6].lower()}\n")

        tot_len, tot_steps = 0, 0


        for _ in range(num_episodes):
            print(f"Episode {self._episode} - ", end="")
            self._game_log[self._episode] = {"reward":[], "action":[], "state":[], "RTG": []}
            while True:
                self._reward = 0
                self._game_main_normal()
                
                if self._map.is_full():
                    print(
                        f"FULL (len: {self._snake.len()} | steps: {self._snake.steps})"
                    )
                    self._reward = 1
                    self._game_log[self._episode]["reward"][-1] = 1
                    break
                if self._snake.dead:
                    print(
                        f"DEAD (len: {self._snake.len()} | steps: {self._snake.steps})"
                    )
                    self._game_log[self._episode]["reward"][-1] = -1
                    self._reward = -1
                    break
                if self._snake.steps >= steps_limit:
                    print(
                        f"STEP LIMIT (len: {self._snake.len()} | steps: {self._snake.steps})"
                    )
                    self._write_logs()  # Write the last step
                    self._game_log[self._episode]["reward"][-1] = -1
                    self._reward = -1
                    break
            tot_len += self._snake.len()
            tot_steps += self._snake.steps       
            self._reset()

        avg_len = tot_len / num_episodes
        avg_steps = tot_steps / num_episodes
        print(
            f"\n[Summary]\nAverage Length: {avg_len:.2f}\nAverage Steps: {avg_steps:.2f}\n"
        )

        self._on_exit()

    def _create_img(self, size=10, mode="human"):
        # size = 10
        # mode = "human"
        board_size = self._conf.map_rows
     

        self.img = np.zeros((board_size*size, board_size*size, 3), dtype=np.uint8)

        # cv2.rectangle(self.img, (   (self._map._food.x- 1) * size,    (self._map._food.y- 1) * size),
        #         (   (self._map._food.x- 1) * size + size,    (self._map._food.y- 1) * size + size), (0, 0, 255), -1,
        #         lineType=cv2.LINE_AA)
        if  self._map._food:
            cv2.circle(
                self.img,(   (self._map._food.x- 1) * size + size//2 ,    (self._map._food.y- 1) * size + size//2 ), size//2, (0, 0, 255), -1, lineType=cv2.LINE_AA
            )
        total_snake_len = len(self.snake.bodies)
        for i, position in enumerate(self.snake.bodies):
            # alpha = self.snakeGame.snake.get_current_snake_value(i)
            alpha = (total_snake_len-i)/total_snake_len*0.9+0.1
            # print(i, alpha)

            cv2.rectangle(self.img, ( (position.x - 1)* size, (position.y - 1) * size),
                        ((position.x - 1)  * size + size, (position.y - 1) * size + size), (0, 255, 0), 1, #2
                        lineType=cv2.LINE_AA)
            overlay = self.img.copy()

            # res = cv2.addWeighted(sub_img, alpha, white_rect, 1-alpha, 1.0)
            cv2.rectangle(overlay, ( (position.x - 1)* size, (position.y - 1) * size),
                            ((position.x - 1)  * size + size, (position.y - 1) * size + size), (0, 255, 0), -1,
                            lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0, self.img)
            # self.img[(position.x - 1)* size:(position.x - 1)  * size + size, (position.y - 1) * size:(position.y - 1) * size + size] = res
        if mode == "human":
            cv2.imshow("Snake", self.img)
            cv2.waitKey(10)
            frame_time = time.time() + 0.1

            while time.time() < frame_time:
                continue
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def _run_dqn_train(self):
        try:
            while not self._game_main_dqn_train():
                pass
        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            self._on_exit()

    def _game_main_dqn_train(self):
        if not self._map.has_food():
            self._map.create_rand_food()

        if self._pause:
            return

        episode_end, learn_end = self._solver.train()

        if episode_end:
            self._reset()

        return learn_end

    def _game_main_normal(self):
        if not self._map.has_food():
            self._reward = 1
            self._map.create_rand_food()

        if self._pause or self._is_episode_end():
            return

        new_direc = self._solver.next_direc()
        self._update_direc(new_direc)

        if self._conf.mode == GameMode.NORMAL and self._snake.direc_next != Direc.NONE:
            self._write_logs()

        img = self._create_img(mode=None)
        self._reward = self._snake.move()

        self._game_log[self._episode]["reward"].append(self._reward)
        self._game_log[self._episode]["RTG"].append(self._reward)
        self._game_log[self._episode]["action"].append(new_direc.value)
        self._game_log[self._episode]["state"].append(img)

        if self._is_episode_end():
            self._write_logs()  # Write the last step

    def _plot_history(self):
        self._solver.plot()

    def _update_direc(self, new_direc):
        # print("new_direc", new_direc)
        # print(self._map)
        # print(self._map._food)
        # print(self.snake.bodies)
        

        
        self._snake.direc_next = new_direc
        # self._reward = 0
        if self._pause:
            self._reward = self._snake.move()

   

        # self._reward = 0

    def _toggle_pause(self):
        self._pause = not self._pause

    def _is_episode_end(self):
        return self._snake.dead or self._map.is_full()

    def _reset(self):
        self._snake.reset()
        self._reward = 0
        
        self._episode += 1
        # self._game_log[self._episode] = {"reward":[], "action":[], "state":[], "RTG": []}

    def _on_exit(self):
        # print(self._game_log)
        
        for ep in self._game_log:
            current_sum = 0
            for i in range(len(self._game_log[ep]["reward"])-1, -1, -1):

                r = self._game_log[ep]["reward"][i]
                current_sum += r
                # print(ep, i, current_sum)
                self._game_log[ep]["RTG"][i] = current_sum
            # print(self._game_log[ep]["RTG"])

        with open(f"{self._conf.game_log_file}/{self._conf.solver_name}.pickle" + "", 'wb') as handle:
            pickle.dump(self._game_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(self._conf.game_log_file, 'wb') as file:
        #      file.write(pickle.dumps(self._game_log)) 
        if self._log_file:
            self._log_file.close()
        if self._solver:
            self._solver.close()

    def _init_log_file(self):
        try:
            os.makedirs("logs")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            self._log_file = None
            self._log_file = open("logs/snake.log", "w", encoding="utf-8")
        except FileNotFoundError:
            if self._log_file:
                self._log_file.close()

    def _write_logs(self):
        self._log_file.write(
            f"[ Episode {self._episode} / Step {self._snake.steps} ]\n"
        )
        for i in range(self._map.num_rows):
            for j in range(self._map.num_cols):
                pos = Pos(i, j)
                t = self._map.point(pos).type
                if t == PointType.EMPTY:
                    self._log_file.write("  ")
                elif t == PointType.WALL:
                    self._log_file.write("# ")
                elif t == PointType.FOOD:
                    self._log_file.write("F ")
                elif (
                    t == PointType.HEAD_L
                    or t == PointType.HEAD_U
                    or t == PointType.HEAD_R
                    or t == PointType.HEAD_D
                ):
                    self._log_file.write("H ")
                elif pos == self._snake.tail():
                    self._log_file.write("T ")
                else:
                    self._log_file.write("B ")
            self._log_file.write("\n")
        self._log_file.write(
            f"[ last/next direc: {self._snake.direc}/{self._snake.direc_next} ]\n"
        )
        self._log_file.write("\n")
