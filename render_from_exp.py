from glob import glob
import torch
from inferno_apps.FaceReconstruction.utils.load import load_model
import os
from skimage.io import imsave


path_to_model = "/home/tien/inferno/assets/" + "FaceReconstruction/models"
model_name = "EMICA-CVT_flame2020_notexture"
face_rec_model, conf = load_model(path_to_model, model_name)
face_rec_model.cuda()
face_rec_model.eval()

from einops import repeat
from skimage.io import imsave



def load_the_defauld_face(path_to_dir):
    cam = torch.load(os.path.join(path_to_dir, "cam.pt"))
    globalpose = torch.load(os.path.join(path_to_dir, "globalpose.pt"))
    lightcode = torch.load(os.path.join(path_to_dir, "lightcode.pt"))
    texcode = torch.load(os.path.join(path_to_dir, "texcode.pt"))
    shapecode = torch.load(os.path.join(path_to_dir, "shapecode.pt"))
    
    return (cam, globalpose, lightcode, texcode, shapecode, )

def read_the_output_tensor(path_to_tensor):
    if isinstance(path_to_tensor, str):
        return torch.load(path_to_tensor, map_location="cpu")
    elif isinstance(path_to_tensor, torch.Tensor):
        return path_to_tensor

def build_the_batch(face_shape_path, prediction_tensor):
    
    expression_and_jawpose = read_the_output_tensor(prediction_tensor)
    
    if isinstance(expression_and_jawpose, dict):
        try:
            expression_and_jawpose = expression_and_jawpose['imgs']
        except:
            expression_and_jawpose = expression_and_jawpose['x']
            
        print(expression_and_jawpose.shape)
        
    if expression_and_jawpose.dim() == 2:
        expression_and_jawpose = expression_and_jawpose.unsqueeze(0)
        print(expression_and_jawpose.shape)

    expression = expression_and_jawpose[:, :, 3:103]
    # expression = reduce(expression, "b c d -> b d", "mean")
    # print(expression.shape)
    # expression = rearrange(expression, "b t d -> b t d")
    
    b, t = expression.shape[0], expression.shape[1]
    
    jaw_pose = expression_and_jawpose[:, :, :3]
    # jaw_pose = reduce(jaw_pose, "b c d -> b d", "mean")
    # jaw_pose = rearrange(jaw_pose, "b t d -> b t d")
    
    defaul_face = load_the_defauld_face(face_shape_path)
    cam, globalpose, lightcode, texcode, shapecode = map(lambda x: torch.mean(x, dim = 0), defaul_face)
    cam, globalpose, lightcode, texcode, shapecode = map(lambda x: repeat(x, "d -> b t d",b=b, t=t).contiguous(), (cam, globalpose, lightcode, texcode, shapecode))
    
    image = torch.rand(b, t, 3, 224, 224)
    
    batch = {
        "image": image,
        "cam": cam,
        "globalpose": globalpose,
        "lightcode": lightcode,
        "texcode": texcode,
        "shapecode": shapecode,
        "expcode": expression,
        "jawpose": jaw_pose,
        "batch": b
    }
    
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        
    return batch


def decode_latent_to_image(face_shape_path, prediction_tensor_path, model=face_rec_model, output_folder=None, name=None, render=False, save_frame=True):
    big_batch = build_the_batch(face_shape_path, prediction_tensor_path)
    
    prediction_batch_size = big_batch.pop('batch', 1)
    # print(prediction_batch_size)
    imgs = []
    
    for batch_idx in range(0, prediction_batch_size):
    
        for frame_idx in range(0, big_batch["expcode"].shape[1], 16):
            
            batch = {key: big_batch[key][batch_idx][frame_idx:frame_idx+16] for key in big_batch}
        
            batch = model.decode_latents(batch, training = False, validation = False, ring_size = 1)
            
            visdict = face_rec_model.visualize_batch(batch, 0, None, in_batch_idx=None)

            current_bs = batch["expcode"].shape[0]
            
            for j in range(current_bs):
                
                img = visdict['shape_image'][j]
                
                imgs.append(img)
                
                if save_frame:
                    os.makedirs(os.path.join(output_dir, basename.split(".")[0], str(batch_idx)), exist_ok=True)
                    
                    imsave(os.path.join(output_dir, basename.split(".")[0],str(batch_idx), f"{frame_idx + j}.jpg"), img)
        
        if render:
            os.system(f"ffmpeg -r 25 -i  {output_dir}/{basename.split('.')[0]}/{str(batch_idx)}/%d.jpg  -vcodec h264 -b:v 10M -y {output_dir}/{basename.split('.')[0]}/{str(batch_idx)}.mp4")
    return imgs

if __name__ == "__main__":

    defaut_face_path = "default_face/"
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--defaut_face_path", type=str, required=False, default=defaut_face_path)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--video_render", type=bool, required=False, default=False)
    
    arg = parser.parse_args()
    
    input_path = arg.input_path
    defaut_face_path = arg.defaut_face_path
    output_dir = arg.output_dir
    video_render = arg.video_render
    
    rendered_images = []

    import os
    
    if "*" in input_path:
        file_path_list = glob(input_path)
        
        # file_path_list = sorted(file_path_list, key = lambda x: int(os.path.basename(x).split(".")[0]))
        
        for file_path in file_path_list:
            
            basename = os.path.basename(file_path)
            os.makedirs(os.path.join(output_dir, basename.split(".")[0]), exist_ok=True)
            decode_latent_to_image(defaut_face_path, file_path, face_rec_model, output_dir, basename, render=video_render)
            # if video_render:
            #     os.system(f"ffmpeg -r 25 -i  {output_dir}/{basename.split('.')[0]}/%d.jpg  -vcodec mpeg4 -b:v 10M -y {output_dir}/{basename.split('.')[0]}.mp4")

    
    else:
        basename = os.path.basename(input_path)
        os.makedirs(os.path.join(output_dir, basename.split(".")[0]), exist_ok=True)
        
        decode_latent_to_image(defaut_face_path, input_path, face_rec_model, output_dir, basename, render=video_render)

        # if video_render:
        #     os.system(f"ffmpeg -r 25 -i  {output_dir}/{basename.split('.')[0]}/%d.jpg  -vcodec mpeg4 -b:v 10M -y {output_dir}/{basename.split('.')[0]}.mp4")
    
