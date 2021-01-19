
from glob import glob
from skimage import exposure, transform

img_shape = (128,128)

# List of paths 
my_glob = glob('./Datasets/images/images*/*.png')
# check if everything was captured (112120)
print(f'Number of Images:  {len(my_glob)}\n')

# Dictionary with key = img_name and value = 'full path'
img_paths = {os.path.basename(x): x for x in my_glob}
# print(img_paths['00025789_001.png'])

# Add path to dataframe
full_df['img_paths'] = full_df['Image Index'].map(img_paths.get)

# Create new directories to contain processed samples
path_list = os.listdir('./Datasets/images/')
for dirs in path_list:
    os.makedirs(f'./Datasets/xray_preprocessed/{dirs}', exist_ok=True)

#---------------------------------------Image Processing ------------------------------------------------------

i=0
for img_path in my_glob :
    i += 1
    
    # print no. of sample after every processed 1000
    if  i % max(1, int(len(my_glob)/1000))==0: print(i, '/', len(my_glob))
        
    # save processed images to xray_preprocessed
    new_path = img_path.replace('images', 'xray_preprocessed')
    img = plt.imread(img_path)
    
    # Increase Exposure with CLAHE
    img = exposure.equalize_adapthist(img, clip_limit=0.05)
    
    # Resize Image to img_shape
    img = transform.resize(img, img_shape, anti_aliasing=True)
    plt.imsave(fname=new_path, arr=img, cmap='gray')
