import random
import  json
import cv2
import os
import glob
import numpy as np
from PIL import Image

OUTPUT_SIZE = (720, 1280)  # Height, Width
height=720
width=1280
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = 'wrong_mask/labels'
IMG_DIR = 'wrong_mask/images'



def main():
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)

    for i in range(100):
        idxs = random.sample(range(len(annos)), 4)
        print(idxs)

        new_image, new_annos = update_image_and_anno(img_paths, annos,idxs,OUTPUT_SIZE, SCALE_RANGE)
        cv2.imwrite('test/output'+str(i)+'.jpg', new_image)
        #print(new_annos)
        f = open("output" + str(i) + ".txt", 'w')
        for ano in new_annos:
            for j in ano:
                f.write(json.dumps(j)+' ')
            f.write("\n")
        f.close()
        for anno in new_annos:
            x_center, y_center, w, h = float(anno[1]) * width, float(anno[2]) * height, float(anno[3]) * width, float(anno[4]) * height
            x1 = round(x_center - w / 2)
            y1 = round(y_center - h / 2)
            x2 = round(x_center + w / 2)
            y2 = round(y_center + h / 2)
            new_image = cv2.rectangle(new_image, (x1,y1), (x2,y2), (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.imwrite('test/output'+str(i)+'.jpg', new_image)

        #new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        #new_image = Image.fromarray(new_image.astype(np.uint8))
        #new_image.show()


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax =  bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * (1 - scale_x)
                ymax = bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
    return output_img, new_anno


def get_dataset(anno_dir, img_dir):

    img_paths = []
    annos = []

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        anno_id = anno_file.split('/')[-1].split('.')[0].split('\\')[-1]

        with open(anno_file, 'r') as f:
            num_of_objs = int(file_len(f.name))
            img_path = os.path.join(img_dir, f'{anno_id}.jpg')
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            del img

            boxes = []
            for _ in range(num_of_objs):
                obj = f.readline().rstrip().split(' ')
                obj = [float(elm) for elm in obj]
                if 3 < obj[0]:
                    continue
                class_id=int(obj[0])

                x1=obj[1]
                y1=obj[2]
                x2=obj[3]
                y2=obj[4]

                boxes.append([class_id, x1, y1, x2, y2])

            if not boxes:
                continue
        img_paths.append(img_path)
        annos.append(boxes)
    return img_paths, annos


if __name__ == '__main__':
    main()
