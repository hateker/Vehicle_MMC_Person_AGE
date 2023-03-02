from openvino.inference_engine import IECore
import cv2
import numpy as np
import time
import glob
import argparse
import os
import pandas

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data',help='path to dataset')
args = parser.parse_args()

class Vehicle_Detector:
    def __init__(self,model_xml,model_bin,device,requests):
        ie = IECore()
        self.net = ie.read_network(model= model_xml, weights= model_bin)
        self.exec_net = ie.load_network(network=self.net, num_requests=requests, device_name=device)
        self.label_map = {0:"car", 1:'lorry', 2:'taxi', 3:'truck'}


        for blob_name in self.net.input_info:
            if len(self.net.input_info[blob_name].input_data.shape) == 4:
                self.input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                    .format(len(self.net.input_info[blob_name].input_data.shape), blob_name))

        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        # mean = (0.485, 0.456, 0.406) ## RGB
        mean = (0.406, 0.456, 0.485)
        # std = (0.229, 0.224, 0.225) ##RGB
        std = (0.225, 0.224, 0.229)

        self.mean = [x * 255 for x in mean]
        self.std = [x* 255 for x in std]
        print(self.mean)
        print(self.std)

        self.feed_dict= {self.input_blob:[self.h, self.w, 1]}
        self.output_name = []
        for blob_name in self.net.outputs:
            self.output_name.append(blob_name)

        self.active_id = []
        self.waiting_id = []

        for request in range(requests):
            self.active_id.append(request)

    def start_infer(self,frame,request_ID):
        try :
            frame_in = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_CUBIC).astype(float)
            # cv2.imshow("in",frame_in)
            # cv2.waitKey(0)
            ## normalize
            # print(frame_in)
            frame_in = (frame_in[:] - self.mean[:]) / self.std[:]
            frame_in = frame_in.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            frame_in = frame_in.reshape((self.n, self.c, self.h, self.w))
            
            # print(frame_in)

            self.feed_dict[self.input_blob] = frame_in
            self.exec_net.start_async(request_id=request_ID, inputs=self.feed_dict)
            self.active_id.remove(request_ID)
            self.waiting_id.append(request_ID)

        except Exception as e:
            print("Error happen at LPD start_infer",e)

    def check_result(self, request_ID):
        if self.exec_net.requests[request_ID].wait(-1) == 0:
            return True
        else:
            return False


    def get_result(self,request_ID):
        res = self.exec_net.requests[request_ID].output_blobs[self.output_name[0]].buffer
        # print(res)
        res = np.argmax(res)
        out = self.label_map[res]
        self.active_id.append(request_ID)
        self.waiting_id.remove(request_ID)
        return out


class simple_classifier():
    def __init__(self):
        self.VD = Vehicle_Detector("/home/ubuntu/Desktop/EfficientNet/pytorch-image-models/model_best.xml",
            "/home/ubuntu/Desktop/EfficientNet/pytorch-image-models/model_best.bin",
            "CPU", 15)
        
    def run(self,frame):
        ID = VD.active_id[0]
        VD.start_infer(frame,ID)
        while VD.check_result(ID) == False:
            pass
        result = VD.get_result(ID)
        return result


VD = Vehicle_Detector("/home/ubuntu/Desktop/EfficientNet/pytorch-image-models/model_best.xml",
                        "/home/ubuntu/Desktop/EfficientNet/pytorch-image-models/model_best.bin",
                        "CPU", 15)
for image in glob.glob("%s/*.jpg" %args.data):
    frame = cv2.imread(image)
    cv2.imshow("ori",frame)
    h, w ,c = frame.shape
    print(frame.shape)
    # frame[:,:,0] = 255
    # frame[:,:,1] = 255
    # frame[:,:,2] = 255
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    cv2.imshow("gaodim",frame)
    cv2.waitKey(0)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # # for coun1, data in enumerate(frame):
    # #     for count2, data1 in enumerate(data):
    # #         # print(frame[coun1][count2])
    # #         frame[coun1][count2]
    # print(frame.shape)
    # # for data in frame:
    # #     print(data)
    # frame = np.zeros_like(frame)
    # cv2.imshow("black",frame)
    # cv2.waitKey(0)
    
    # dark_arg_list = np.where(frame[:,:,1] >= 130)[0]
    # print(dark_arg_list)
    # print(frame[dark_arg_list])
    # cv2.imshow("gg",frame[dark_arg_list])
    # cv2.waitKey(0)
    # for data in frame:
    #     print(data)
    # pd = pandas.DataFrame(frame[0])
    # pd.to_csv("kek.csv")
    # print(pd)
    
    # np.savetxt("chrlse.csv",frame, delimiter= ",")
    # ID = VD.active_id[0]
    # VD.start_infer(frame,ID)
    # while VD.check_result(ID) == False:
    #     pass
    # result = VD.get_result(ID)
    # print(result)

    # if result not in  os.listdir():
    #     os.mkdir(result)
    # cv2.imwrite("%s/%s" %(result,image.split("/")[-1]), frame)
    break


