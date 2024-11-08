
import os
from preprocess.extract_pspi import get_pspi_from_video
import torch

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract PSPI from a list of video')
    
    parser.add_argument("--input_path", type=str, default='output/stimuli_*_expressiveness_*_emotion_*/0')
    
    parser.add_argument("--output_dir", type=str, default="output/pspi")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    
    if "*" in input_path:
        import glob
        video_paths = glob.glob(input_path)
        
        for video_path in video_paths:
            
            # .../video_name/0/
            video_name = video_path.split("/")[-2]
            
            pspi = get_pspi_from_video(video_path)
            
            torch.save(pspi, os.path.join(output_dir, "{}.pt".format(video_name)))
            
    else:
        
        video_name = os.path.basename(input_path)
        
        pspi = get_pspi_from_video(input_path)
        
        torch.save(pspi, os.path.join(output_dir, "{}.pt".format(video_name)))