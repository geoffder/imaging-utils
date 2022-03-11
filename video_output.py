import cv2

def array_to_mp4(arr, path, fps=60, scale=5):
    arr = upscale(normalize_uint8(arr), scale, axes=[1, 2])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, fps, arr.shape[1:])
    for frame in arr:
        img = np.tile(frame.reshape(*frame.shape, 1), (1, 1, 3))
        video.write(img)

    video.release()
