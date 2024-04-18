import pickle
import matplotlib.pyplot as plt



with open('data/WeakGreedySolver.pickle', 'rb') as file:
     game_log = pickle.load(file)
total_length = 0
for key in game_log:
    # print(game_log[key]["reward"])
    if len(game_log[key]["action"]) == 0:
         continue
    print("Episode: ", key)
    print("action:", len(game_log[key]["action"]))
    print("reward:",len(game_log[key]["reward"]))
    print("RTG:",len(game_log[key]["RTG"]))
    print("reward:",game_log[key]["reward"])
    print("RTG:",game_log[key]["RTG"])
    print("reward 1: ", len([ r for r in game_log[key]["reward"] if r ==1]))
    print("state:",len(game_log[key]["state"]))
    print(game_log[key]["state"][0].shape)
    total_length += len(game_log[key]["action"])
    print()
print("Total Len:", total_length)
# print(game_log[1]["state"][0])
# plt.imshow(game_log[1]["state"][0])
# plt.show()