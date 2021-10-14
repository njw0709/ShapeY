import subprocess


def run_script(command, **kwargs):
    for k in kwargs.keys():
        command = command + ' ' + '--' + str(k) + ' ' + str(kwargs[k])
    command = command.split(' ')
    print(command)
    subprocess.run(command)
    print('Done with ' + str(command))

