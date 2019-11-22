import numpy as np

def read_box(file):
    num_images = 0
    num_no_ground_truth = 0
    num_no_detected_face = 0

    GT_bboxes = []
    GT_bboxes_principle = []
    Pre_bboxes = []
    Pre_bboxes_principle = []
    GT_bboxes_images=[]
    Pre_bboxes_images = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:

            num_images += 1
            line = str.split(line, ' ')
            img_file = line[0]
            ## read the ground truth of bounding box
            gt_num = int(line[1])
            if not gt_num:
                print('===============>>>> No groud truth of bounding box in %s!'%img_file)
                num_no_ground_truth += 1
                continue
            gt_bboxes = np.zeros((gt_num, 4))

            ## read groud truth of bounding box
            for i in range(gt_num):
                i_start = 2+i*4
                i_end = 2+(i+1)*4
                gt_bboxes[i] = list(map(int, line[i_start:i_end]))

            ## chose the principal face photo on id card
            if gt_num>1:
                boxes_area = np.zeros(gt_num)
                for i in range(gt_num):
                    area =  abs((gt_bboxes[i][0]-gt_bboxes[i][2])*(gt_bboxes[i][1]-gt_bboxes[i][3]))
                    boxes_area[i] = area
                idx_principal = np.argmax(boxes_area)
                GT_bboxes_principle.append(gt_bboxes[idx_principal])
            else:
                GT_bboxes_principle.append(gt_bboxes[0])

            gt_idx_end = i_end
            GT_bboxes.append(gt_bboxes)
            GT_bboxes_images.append(img_file)


            ## read the predicted bounding box and the confidences
            pre_num = (len(line)-gt_idx_end)/5
            if pre_num == 0:
                print('%s No face has been detected!'%img_file)
                num_no_detected_face += 1
                Pre_bboxes.append([])
                Pre_bboxes_images.append(img_file)
            else:
                pre_bboxes = np.zeros((pre_num, 5))
                for i in range(pre_num):
                    i_start = gt_idx_end + i * 5
                    i_end = gt_idx_end + (i + 1) * 5
                    pre_bboxes[i, 0:4] = list(map(int, line[i_start:i_end-1]))
                    ## the last value is the confidence of each predicted bounding box
                    pre_bboxes[i, 4] = float(line[i_end-1])
                Pre_bboxes.append(pre_bboxes)
                Pre_bboxes_images.append(img_file)
            ## chose the principal face photo on id card
            # if pre_num>1:
            #     boxes_area = np.zeros(pre_num)
            #     for i in range(pre_num):
            #         area =  abs((pre_bboxes[i][0]-pre_bboxes[i][2])*(pre_bboxes[i][1]-pre_bboxes[i][3]))
            #         boxes_area[i] = area
            #     idx_principal = np.argmax(boxes_area)
            #     Pre_bboxes_principle.append(pre_bboxes[idx_principal])
            # else:
            #     Pre_bboxes_principle.append(pre_bboxes[0])

        print('Total %d images, %d No ground truth, %d No detected face!'%(num_images, num_no_ground_truth, num_no_detected_face))

    return GT_bboxes_principle, GT_bboxes, Pre_bboxes, GT_bboxes_images, Pre_bboxes_images