import cv2
import time
import numpy as np
from random import randint
import os

# image1 = cv2.imread("group.jpg")



protoFile = "C:\\action_data\\pose\\coco\\pose_deploy_linevec.prototxt"
weightsFile = "C:\\action_data\\pose\\coco\\pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    # mapMask = np.uint8(mapSmooth>threshold)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []


    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    else:
        print("---  There is this folder!  ---")




input = 'C:\\action_data\\KTH\\video'
txt_directory = 'C:\\action_data\\KTH\\txt'
output_path = 'C:\\action_data\\KTH\\output'

for (direcpath, direcnames, dirfiles) in os.walk(input):
    for direcname in direcnames:
        p = os.path.join(input, direcname)
        print('p',p)
        for (dirpath, dirnames, files) in os.walk(p):
            files.sort()
            print(len(files))
            s = 0
            for file in files:
                input_source = os.path.join(p,file)
                print('input_source', input_source)
                dir_output = os.path.join(output_path, direcname)
                print('dir_output', dir_output)
                mkdir(dir_output)
                video_output = os.path.join(output_path, direcname,file)
                print('output_video', video_output)
                s +=1
                if s < 10:
                    csv_path = os.path.join(txt_directory, direcname,'0'+str(s))
                elif s >= 10:
                    csv_path = os.path.join(txt_directory, direcname, str(s))
                print('txt', csv_path)
                mkdir(csv_path)

                # input_source = "/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf action/Golf-Swing-Front/007/RF1-13588_70046.avi"
                # csv_path = '/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf action/Golf-Swing-Front/golf_swing_front/007da/'
                cap = cv2.VideoCapture(input_source)
                # cap = cv2.VideoCapture(0)
                hasFrame, frame = cap.read()

                vid_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                             (frame.shape[1], frame.shape[0]))

                net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

                detected_keypoints_all = []


                # while cv2.waitKey(1) < 0:
                num = 0
                ci = 0
                while (cap.isOpened()):

                    t = time.time()
                    ci += 1
                    hasFrame, frame = cap.read()
                    frameCopy = np.copy(frame)
                    print('ci',ci)
                    if hasFrame == True:
                        if ci % 2 == 0:
                            frameWidth = frame.shape[1]
                            frameHeight = frame.shape[0]

                            t = time.time()
                            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

                            # Fix the input Height and get the width according to the Aspect Ratio
                            inHeight = 368
                            inWidth = int((inHeight/frameHeight)*frameWidth)

                            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                                      (0, 0, 0), swapRB=False, crop=False)

                            net.setInput(inpBlob)
                            output = net.forward()

                            # print("Time Taken in forward pass = {}".format(time.time() - t))

                            detected_keypoints = []
                            detected_keypoints_raw = []
                            keypoints_list = np.zeros((0,3))
                            keypoint_id = 0
                            threshold = 0.1
                            keypoints_2D = np.zeros(2)
                            # print(keypoints_2D)

                            keypoint_man_1 = []
                            for part in range(nPoints):
                                probMap = output[0,part,:,:]
                                probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
                                keypoints = getKeypoints(probMap, threshold)
                                if keypoints:
                                    detected_keypoints_raw.append((keypoints[0][0],keypoints[0][1]))
                                else:
                                    detected_keypoints_raw.append(keypoints_2D)
                                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                                keypoints_with_id = []
                                for i in range(len(keypoints)):
                                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                                    keypoint_id += 1
                                detected_keypoints.append(keypoints_with_id)

                            # print('detedted_keypoints_raw', np.array(detected_keypoints_raw).shape, detected_keypoints_raw)

                            if num < 10:
                                np.savetxt(csv_path + '/0'+str(num) + '.csv', detected_keypoints_raw)
                            elif num >= 10:
                                np.savetxt(csv_path + '/'+str(num) + '.csv', detected_keypoints_raw)




                            # np.savetxt(csv_path + str(00) + '.csv', detected_keypoints_all)


                            num += 1
                            detected_keypoints_all.append(detected_keypoints_raw)

                            # print('detedted_keypoints_all', np.array(detected_keypoints_all).shape)


                            frameClone = frame.copy()
                            for i in range(nPoints):
                                for j in range(len(detected_keypoints[i])):
                                    # 取18行中i行， 然后取第j列的点，给每个部位上色
                                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, colors[i], -1, cv2.LINE_AA)
                            # cv2.imshow("Keypoints",frameClone)


                            valid_pairs, invalid_pairs = getValidPairs(output)
                            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)


                            for i in range(17):
                                for n in range(len(personwiseKeypoints)):
                                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                    if -1 in index:
                                        continue
                                    B = np.int32(keypoints_list[index.astype(int), 0])
                                    A = np.int32(keypoints_list[index.astype(int), 1])
                                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                            vid_writer.write(frameClone)
                            # cv2.imshow("Detected Pose", frameClone)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    else:
                        break

                        # Release everything if job is finished
                cap.release()
                vid_writer.release()
                cv2.destroyAllWindows()


                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         # 退出while循环，退出程序
                    #         break
                    # cv2.destroyAllWindows()


                    # vid_writer.release()
                    # cv2.DestroyWindow("Detected Pose")




                #
