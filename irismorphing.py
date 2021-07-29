import os
import numpy as np
import cv2
import math
import locator
import aligner
import warper
import blender
import plotter

def load_image_points(path, size):
  img = cv2.imread(path)
  points = locator.face_points(img)

  if len(points) == 0:
    print('No iris in %s' % path)
    return None, None
  else:
    return aligner.resize_align(img, points, size)

def load_valid_image_points(imgpaths, size):
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      print(path)
      yield (img, points)

def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)

def morph(src_img, src_points, dest_img, dest_points,
          video, width=500, height=600, num_frames=20, fps=10,
          out_frames=None, plot=False, background='black'):

  size = (height, width)
  stall_frames = np.clip(int(fps*0.15), 1, fps)  # Show first & last longer
  plt = plotter.Plotter(plot, num_images=num_frames, out_folder=out_frames)
  num_frames -= (stall_frames * 2)  # No need to process src and dest image

  plt.plot_one(src_img)
  video.write(src_img, 1)

  # Produce morph frames!
  for percent in np.linspace(1, 0, num=num_frames):
    points = locator.weighted_average_points(src_points, dest_points, percent)
    src_iris = warper.warp_image(src_img, src_points, points, size)
    end_iris = warper.warp_image(dest_img, dest_points, points, size)
    average_iris = blender.weighted_average(src_iris, end_iris, percent)

    if background in ('transparent', 'average'):
      mask = blender.mask_from_points(average_iris.shape[:2], points)
      average_iris = np.dstack((average_iris, mask))

      if background == 'average':
        average_background = blender.weighted_average(src_img, dest_img, percent)
        average_iris = blender.overlay_image(average_iris, mask, average_background)

    plt.plot_one(average_iris)
    plt.save(average_iris)
    video.write(average_iris)

  plt.plot_one(dest_img)
  video.write(dest_img, stall_frames)
  plt.show()

def getPoints(img, circle):

  points = [[0, 0], [(img.shape[1] - 1), 0], [0, (img.shape[0] - 1)],
                [(img.shape[1] - 1), (img.shape[0] - 1)]]
  xc = circle[0]  # x-co of circle (center)
  yc = circle[1]  # y-co of circle (center)
  r = circle[2]  # radius of circle
  for i in range(0, 360, 10):
    x = xc + r * math.cos(math.radians(i))
    y = yc + r * math.sin(math.radians(i))
    x = int(x)
    y = int(y)
    points.append([x, y])

  xc = circle[3]  # x-co of circle (center)
  yc = circle[4]  # y-co of circle (center)
  r = circle[5]  # radius of circle
  for i in range(0, 360, 10):
    x = xc + r * math.cos(math.radians(i))
    y = yc + r * math.sin(math.radians(i))
    x = int(x)
    y = int(y)
    # Create array with all the x-co and y-co of the circle
    points.append([x, y])
  points= np.array(points)
  return points


def morphIris(imageNames, circles, filename, background='average'):

  # Reading of the two component images
  src_img =cv2.imread(imageNames[0])
  dest_img = cv2.imread(imageNames[1])

  # Determining landmark points from two images
  src_points = getPoints(src_img, circles[0])
  dest_points = getPoints(dest_img, circles[1])

  size = (src_img.shape[0],src_img.shape[1])
  percent = 0.5 # blending factor

  # Weighted average of corresponding triangles from both images
  points = locator.weighted_average_points(src_points, dest_points, percent)

  # Warping of two images
  src_iris = warper.warp_image(src_img, src_points, points, size)
  end_iris = warper.warp_image(dest_img, dest_points, points, size)

  # Blending of two images
  average_iris = blender.weighted_average(src_iris, end_iris, percent)

  if background in ('transparent', 'average'):
    mask = blender.mask_from_points(average_iris.shape[:2], points)
    average_iris = np.dstack((average_iris, mask))

    if background == 'average':
      average_background = blender.weighted_average(src_img, dest_img, percent)
      average_iris = blender.overlay_image(average_iris, mask, average_background)

  cv2.imwrite(filename,average_iris)

if __name__ == "__main__":
    # Path of two component images
    image1Path = 'Images/Image01.bmp'
    image2Path =  'Images/Image02.bmp'

    # Segmentation information of two compoment images in the form on [iriscenterX, iriscenterY, irisradius, pupilcenterX, pupilcenterY, pupilradius]
    segInfo1 = [130,142,102,130,144,45]
    segInfo2 = [153,119,105,156,118,52]

   # Path of morphed image
    morphImagePath = 'Images/Morphed_Image.bmp'

    imagesPath =[image1Path, image2Path]
    circles = [segInfo1, segInfo2]

    try:
      morphIris(imagesPath, circles, morphImagePath)
    except Exception as ex:
      print(ex)


