import subprocess
import functools
from itertools import combinations


def run_script(command, **kwargs):
    for k in kwargs.keys():
        command = command + ' ' + '--' + str(k) + ' ' + str(kwargs[k])
    command = command.split(' ')
    print(command)
    subprocess.run(command)
    print('Done with ' + str(command))

def check_image_order(user_img_list: list, reference_img_list: list) -> bool:
    return True

def make_axis_of_interest() -> list:
    axes = ['x','y','p','r','w']
    axis_of_interest = []
    for choose in range(1,4):
        for comb in combinations(axes, choose):
            axis_of_interest.append(functools.reduce(lambda a, b: a + b, comb))
    axis_of_interest.sort()
    return axis_of_interest