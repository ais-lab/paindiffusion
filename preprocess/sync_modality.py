video_fps = 25
bio_signal_freq = 512
temperature_freq = 32
label_freq = 32


def convert_id_to_time(id, sample_rate=25, ms=False):
    if ms:
        return id / sample_rate * 1000
    return id / sample_rate


def convert_time_to_id(time, sample_rate=25):
    return int(time * sample_rate)


def convert_frame_id_to_temperature_id(frame_id, video_fps=25, temperature_freq=32):
    time = convert_id_to_time(frame_id, video_fps)
    return convert_time_to_id(time, temperature_freq)


def convert_temperature_id_to_frame_id(
    temperature_id, video_fps=25, temperature_freq=32
):
    time = convert_id_to_time(temperature_id, temperature_freq)
    return convert_time_to_id(time, video_fps)


def convert_frame_id_to_label_id(frame_id, video_fps=25, label_freq=32):
    time = convert_id_to_time(frame_id, video_fps)
    return convert_time_to_id(time, label_freq)
