import argparse
import os
import shutil
import time

def run_background():

    i = 1
    while True:
        print('loop %d times' % i)

        files = os.listdir(source_dir)
        if len(files) == 0:
            time.sleep(5)
        else:
            for file in files:
                source_path = os.path.join(source_dir, file)
                des_path = os.path.join(des_dir, file)
                shutil.move(source_path, des_path)
        i += 1
        if (i >= 20):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--source_dir', type=str, default="E://tmp//source_dir",
        help='path to model weight file'
    )

    parser.add_argument(
        '--des_dir', type=str, default="E://tmp//des_dir",
        help='The location of the model checkpoint files'
    )

    FLAGS = parser.parse_args()

    source_dir = FLAGS.source_dir
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    os.mkdir(source_dir)

    des_dir = FLAGS.des_dir
    if os.path.exists(des_dir):
        shutil.rmtree(des_dir)
    os.mkdir(des_dir)

    run_background()