from deep_rlsp.util.mujoco import initialize_mujoco_from_obs


def save_video(ims, filename, fps=20.0):
    import cv2

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        writer.write(im)
    writer.release()


def render_mujoco_from_obs(env, obs, **kwargs):
    env = initialize_mujoco_from_obs(env, obs)
    rgb = env.render(mode="rgb_array", **kwargs)
    return rgb
