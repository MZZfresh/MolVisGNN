import os
import numpy as np
from PIL import Image
from pymol import cmd,util
import zipfile


def process_sdf_file(sdf_file, sdf_output_folder, compressed_npy_folder, rotation_angles):
    # 加载 SDF 文件
    try:
        cmd.load(sdf_file, "molecule")
    except Exception as e:
        print(f"Failed to load {sdf_file}: {e}")
        return None

    cmd.show("sticks", "all")  
    cmd.set("stick_radius", 0.15)  
    cmd.set("stick_quality", 4)  
    cmd.set("stick_ball", "on")  



    util.cba(13,"all",_self=cmd)
    


    screenshots = []


    for i, (x_angle, y_angle) in enumerate(rotation_angles):
        try:
            # 设置渲染样式

            cmd.rotate("x", x_angle)  
            cmd.rotate("y", y_angle)  

            # 截图保存为 PNG
            screenshot_path = os.path.join(sdf_output_folder, f"screenshot_{i}.png")
            cmd.png(screenshot_path, width=512, height=512, dpi=900, ray=True)  
            screenshots.append(screenshot_path)
        
        except Exception as e:
            print(f"Failed to capture screenshot for {sdf_file} at angle {i}: {e}")
            return None  
    cmd.delete("molecule")


    screenshot_arrays = []
    for screenshot in screenshots:
        try:
            img = Image.open(screenshot)
            img_array = np.array(img)
            img_array = np.transpose(img_array, (2, 0, 1))
            screenshot_arrays.append(img_array)
        except Exception as e:
            print(f"Failed to process screenshot {screenshot}: {e}")
            return None  
    

    stacked_screenshot = np.stack(screenshot_arrays, axis=0)  


    npy_file = os.path.join(compressed_npy_folder, f"{os.path.basename(sdf_file).split('.')[0]}.npy")
    np.save(npy_file, stacked_screenshot)

    return stacked_screenshot


def process_all_sdf_files(sdf_folder, output_folder, compressed_npy_folder, rotation_angles):
    failed_files = []  
    

    os.makedirs(compressed_npy_folder, exist_ok=True)
    
    for sdf_file in os.listdir(sdf_folder):
        if sdf_file.endswith(".sdf"):
            sdf_path = os.path.join(sdf_folder, sdf_file)
            sdf_output_folder = os.path.join(output_folder, sdf_file.split('.')[0])  
            os.makedirs(sdf_output_folder, exist_ok=True)
            

            stacked_image = process_sdf_file(sdf_path, sdf_output_folder, compressed_npy_folder, rotation_angles)
            
            if stacked_image is None:
                print(f"Failed to process {sdf_file}")
                failed_files.append(sdf_file)  # 记录失败的文件名

    # 输出所有失败的文件名
    if failed_files:
        print("Failed to process the following files:")
        for failed_file in failed_files:
            print(failed_file)
    else:
        print("All files processed successfully.")


# 示例使用
sdf_folder = "drug_3d"  # input SDF file
output_folder = "drug_3dpng"  # png file
compressed_npy_folder = "npy"  # npy file
rotation_angles = [
    (0, 0),       # Front
    (0, 180),     # Back
    (0, -90),     # Left
    (0, 90),      # Right
    (-90, 0),     # Top
    (90, 0)       # Bottom
]

process_all_sdf_files(sdf_folder, output_folder, compressed_npy_folder, rotation_angles)
