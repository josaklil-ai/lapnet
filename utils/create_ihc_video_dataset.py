import glob
import os
import cv2
import imutils
import pandas as pd 


def _write_video_frames(video_root_dir, video_rgb_out_dir):
    """
    Writes frames from each video in video_root_dir that have not already been written.
    """
    print('Begin to write video frames...')

    all_video_paths = glob.glob(os.path.join(video_root_dir, 'temp_anns/*'))
    all_video_paths = [sub.replace('temp_anns', 'video_dataset/rgb')[:-4] for sub in all_video_paths]
    
    existing_video_paths = glob.glob(os.path.join(video_rgb_out_dir, '*'))

    print(f'{len(all_video_paths)} videos that have complete temporal annotations.')
    print(f'{len(all_video_paths) - len(existing_video_paths)} still left to write.')

    for case in all_video_paths:
        if case in existing_video_paths:
            continue
        else:
            case = glob.glob(os.path.join(video_root_dir, f'**/{os.path.basename(case)}.mp4'))[0]

            # Make folder for video if it doesn't exist
            os.makedirs(os.path.join(video_rgb_out_dir, os.path.basename(case)[:-4]), exist_ok=True)
            print('Writing frames for video: ', case)

            # Read this specific video
            vidcap = cv2.VideoCapture(case)
            fps = vidcap.get(cv2.CAP_PROP_FPS)

            # Write frames of video in correct format to the right folder
            frame_num = 0
            while True:
                success, image = vidcap.read()
                if success:
                    fn = '{:09d}'.format(frame_num)
                    frame = imutils.resize(image, height=240)
                    frame_name = os.path.join(video_rgb_out_dir, os.path.basename(case)[:-4], f'img_{fn}.jpg')

                    cv2.imwrite(frame_name, frame)

                    frame_num += 1
                    if frame_num % 10000 == 0:
                        print(f'Wrote frame number {frame_num}')

                else:
                    break


if __name__ == '__main__':
    video_root_dir = '/pasteur/data/intermountain'
    video_rgb_out_dir = '/pasteur/data/intermountain/video_dataset/rgb'

    _write_video_frames(video_root_dir, video_rgb_out_dir)

    print(f'Finished writing all frames.')