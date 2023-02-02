
# Decorator for iterating over every video
def vid_iterator(root_dir, action_fn, video_fn):
    # Wrapper function called by decorator
    def wrapper():
        # Get all actions/gestures names
        actions = os.listdir(root_dir)
        for action in actions:
            # Function all on each action/gesture
            action_fn(action)

            # Get all filenames for augmentating each
            videos = os.listdir(f"{root_dir}/{action}")

            # Augment video by video
            for video in videos:
                # Function call on each video
                video_fn(video)

    return wrapper

# Decorator for providing runtime of script
def time_decorator(main_fn, output_msg):
    # Wrapper function called by decorator
    def wrapper():
        start_time = time.time()

        # Script method call
        main_fn()

        end_time = time.time()
        print(output_msg, end_time - start_time)

    return wrapper

# Save augmented video as mp4 file
# save(aug_vid, f"{DATA_PATH}/{curr_folder}/aug_{aug_num}.mp4", fps, (width, height))
def save_video(vid_frames, vid_dest, vid_fps, vid_dim):
    # Define codec and video writer (Four character code for uniquely identifying file formats)
    fourcc = 'mp4v'
    video_writer = cv2.VideoWriter_fourcc(*fourcc)

    # Save video 
    out = cv2.VideoWriter(vid_dest, video_writer, vid_fps, vid_dim)
    for frame in vid_frames:
        out.write(frame)
    out.release()
