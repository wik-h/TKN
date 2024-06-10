import os
import pandas as pd
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image

excel_file = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\vedio_frames.xlsx'
sheet_name = 'Sheet1'

# 读取Excel数据
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 规范化文件夹路径
base_folder = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch'
frame_folder = os.path.normpath(os.path.join(base_folder, 'frame'))
output_folder = os.path.normpath(os.path.join(base_folder, 'video_blocks'))

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'已创建输出文件夹: {output_folder}')
else:
    print(f'输出文件夹 {output_folder} 已存在')

# 遍历DataFrame，提取并合成视频块
for idx, row in df.iterrows():
    start_frame = row['起始帧号']
    end_frame = row['终止帧号']
    behavior = row['行为']

    # 初始化视频块的图像列表
    frames = []

    for frame_num in range(start_frame, end_frame + 1):
        # 读取帧文件
        frame_path = os.path.join(frame_folder, f'{frame_num}.jpg')
        frame = Image.open(frame_path)
        frame_np = np.array(frame)  # 转换为NumPy数组
        frames.append(frame_np)

    # 合成视频块
    video_block = ImageSequenceClip(frames, fps=30)

    # 保存合成的视频块
    if behavior == 1:
        video_block_path = os.path.join(output_folder, f'grab_video_block_{idx}.mp4')
    else:
        video_block_path = os.path.join(output_folder, f'non_grab_video_block_{idx}.mp4')

    video_block.write_videofile(video_block_path, codec='libx264', fps=30)

    print(f'视频块 {idx} 已生成：{video_block_path}')