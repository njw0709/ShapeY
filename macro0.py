import subprocess
import os

python_version = "python3.6"

#runs on base shapey
networks = [
    # 'tf_efficientnet_l2_ns_475', 
    # 'resnetv2_101x3_bitm', 
    # 'mixer_b16_224',
    # 'resnetv2_101x3_bitm_in21k',
    # # 'simclr_v1_resnet50x4', #needs to be re-run
    # 'vit_large_patch16_224',
    'example'
]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    subprocess.run([python_version,
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR), 
                    'data=ShapeY200_objr_bgdg'])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR),
                    'graph=obj,cat'])

    #Contrast Reversed
    subprocess.run([python_version,
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200_objb_bgdy',
                    'data.project_dir={}'.format(PROJECT_DIR)])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200CR',
                    'data.project_dir={}'.format(PROJECT_DIR),
                    'graph=obj_cr_hard,obj_cr_soft,cat_cr_hard,cat_cr_soft'])