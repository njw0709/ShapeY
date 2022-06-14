import subprocess
import os

python_version = "python3.6"

#runs on base shapey
networks = [
    'simclr_v1_resnet50x1',
    'mixer_b16_224',
    'resnetv2_50x1_bitm',
    'resnetv2_50x1_bit_distilled',
    'vit_large_patch16_224'
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    #Contrast Reversed
    subprocess.run([python_version,
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200CR',
                    'data.project_dir={}'.format(PROJECT_DIR)])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200CR',
                    'data.project_dir={}'.format(PROJECT_DIR),
                    'graph=obj_cr_hard,obj_cr_soft,cat_cr_hard,cat_cr_soft'])