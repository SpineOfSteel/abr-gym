# @title loader
# load_
import json
import os
from typing import Dict, List
import zipfile,tempfile, shutil


BITS_IN_BYTE = 8.0

def load_trace(cooked_trace_path):
    all_cooked_time: List[List[float]] = []
    all_cooked_bw: List[List[float]] = []
    all_file_names: List[str] = []
    temp_dir = None
    
    if zipfile.is_zipfile(cooked_trace_path):
        temp_dir = tempfile.mkdtemp()
        print(temp_dir)
        with zipfile.ZipFile(cooked_trace_path, 'r') as zip_ref:            
            zip_ref.extractall(temp_dir)
        source_dir = temp_dir + '/train' #MARK
        
    else:
        source_dir = cooked_trace_path

    if not os.path.isdir(source_dir):
        raise ValueError(f"Trace path '{cooked_trace_path}' is neither a valid directory nor a zip file.")

    cooked_files = sorted(os.listdir(source_dir))
    #print(os.listdir(source_dir))
    for cooked_file in cooked_files:
        file_path = os.path.join(source_dir, cooked_file)
        if not os.path.isfile(file_path):
            continue

        cooked_time: List[float] = []
        cooked_bw: List[float] = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                if len(parse) != 2 :
                    print(f"Invalid line: {line}")
                    continue
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))

        if cooked_time and cooked_bw:
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)

    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir) # Clean up temporary directory

    return all_cooked_time, all_cooked_bw, all_file_names

def load_video_size( video_metadata_file, return_unit: str = 'bytes'):
    with open(video_metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segment_duration_ms = int(data.get('segment_duration_ms', 4000))
    bitrates_kbps = list(data['bitrates_kbps'])
    segment_sizes_bits = data['segment_sizes_bits']

    if not segment_sizes_bits:
        raise ValueError('segment_sizes_bits must be non-empty')

    bitrate_levels = len(bitrates_kbps)
    total_video_chunk = len(segment_sizes_bits)
    video_size: Dict[int, List[int]] = {i: [] for i in range(bitrate_levels)}

    for seg_idx, seg_sizes in enumerate(segment_sizes_bits):
        if len(seg_sizes) != bitrate_levels:
            raise ValueError(
                f'segment_sizes_bits[{seg_idx}] has {len(seg_sizes)} entries; expected {bitrate_levels}'
            )
        for bitrate_idx, size_bits in enumerate(seg_sizes):
            size_bits = int(size_bits)
            if return_unit == 'bits':
                size_value = size_bits
            elif return_unit == 'bytes':
                size_value = int(round(size_bits / BITS_IN_BYTE))
            else:
                raise ValueError("return_unit must be 'bits' or 'bytes'")
            video_size[bitrate_idx].append(size_value)

    metadata = {
        'segment_duration_ms': segment_duration_ms,
        'bitrates_kbps': bitrates_kbps,
        'bitrate_levels': bitrate_levels,
        'total_video_chunk': total_video_chunk,
        'size_unit': return_unit,
    }
    return video_size, metadata


'''
all_cooked_time, all_cooked_bw, all_file_names = load_trace('/content/drive/MyDrive/abr-gym/train.zip')
video_size, meta = load_video_size('/content/drive/MyDrive/abr-gym/envivio_48.json', return_unit="bytes")
env = Environment(
    all_cooked_time,
    all_cooked_bw,
    video_size,
    config=EnvConfig()
)



print(f'Loaded {len(all_file_names)} traces  bw: {sum( [len(i) for i in all_cooked_bw])}    time: {all_cooked_bw[0][0]}  ')
print(f'FCC trace file: {all_file_names[54] if all_file_names else "N/A"}')
print(f'Norway trace file: {all_file_names[0] if all_file_names else "N/A"}')

print(env.cfg())
print(meta)
'''