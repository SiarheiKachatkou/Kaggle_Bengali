import os
import subprocess

def save(save_fn, dst_path_may_be_gs):
    tmp_name='tmp'
    save_fn(tmp_name)
    if dst_path_may_be_gs.startswith('gs:'):
        subprocess.check_call(
              ['gsutil',  'cp', tmp_name, dst_path_may_be_gs])
    else:
        os.rename(tmp_name, dst_path_may_be_gs)
