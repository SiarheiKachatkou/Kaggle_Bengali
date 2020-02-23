import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--job-dir',
    type=str,
    required=True,
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train_bin_files_dir',
    type=str,
    required=True,
    help='Training files dir local or GCS')

  parser.add_argument(
    '--test_bin_files_dir',
    type=str,
    required=True,
    help='Test files dir local or GCS')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')

  parser.add_argument('--ckpt_name',type=str,default='')

  return parser.parse_args()
