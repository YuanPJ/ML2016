from PIL import Image
import sys

im1 = Image.open(sys.argv[1])
im2 = im1.rotate(180)
im2.save("ans2.png")

