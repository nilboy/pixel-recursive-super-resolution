import numpy as np
from skimage.io import imsave

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference
def save_samples(np_imgs, img_path):
  """
  Args:
    np_imgs: [N, H, W, 3] float32
    img_path: str
  """
  np_imgs = np_imgs.astype(np.uint8)
  N, H, W, _ = np_imgs.shape
  num = int(N ** (0.5))
  merge_img = np.zeros((num * H, num * W, 3), dtype=np.uint8)
  for i in range(num):
    for j in range(num):
      merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]

  imsave(img_path, merge_img)
def logits_2_pixel_value(logits, mu=1.1):
  """
  Args:
    logits: [n, 256] float32
    mu    : float32
  Returns:
    pixels: [n] float32
  """
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, 256, dtype=np.float32)
  pixels = np.sum(probs * pixel_dict, axis=1)
  return np.floor(pixels)

