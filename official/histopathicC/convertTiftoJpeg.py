import os
from PIL import Image
# for f in train/*.tif; do rm "$f"; done

current_path = '/path/to/images'
for root, dirs, files in os.walk(current_path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        # if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % name)
            # If a jpeg with the name does *NOT* exist, covert one from the tif.
            else:
                outputfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print("Converting jpeg for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outputfile, "JPEG", quality=100)
                except Exception as e:
                    print(e)
