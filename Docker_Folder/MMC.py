from openvino.inference_engine import IECore
import cv2
import numpy as np

class MM:
    def __init__(self, model_xml, model_bin, device="CPU", requests=2):
        ie = IECore()
        self.net = ie.read_network(model= model_xml, weights= model_bin)
        self.exec_net = ie.load_network(network=self.net, num_requests=requests, device_name=device)

        # self.callback = lambda status, py_data : outsource_callback(py_data)
        self.label_map = {
            "0":"HONDA_ACCORD", 
            "1":"HONDA_CIVIC", 
            "2":"HONDA_CRV", 
            "3":"HYUNDAI_ELANTRA", 
            "4":"HYUNDAI_SONATA", 
            "5":"HYUNDAI_TUCSON", 
            "6":"MAZDA_2", 
            "7":"MAZDA_3", 
            "8":"MAZDA_5", 
            "9":"TOYOTA_CAMRY", 
            "10":"TOYOTA_HILUX", 
            "11":"TOYOTA_YARIS"
        }


        for blob_name in self.net.input_info:
            if len(self.net.input_info[blob_name].input_data.shape) == 4:
                self.input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(self.net.input_info[blob_name].input_data.shape), blob_name))

        self.output_name = []
        for blob_name in self.net.outputs:
            self.output_name.append(blob_name)

        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        self.feed_dict= {self.input_blob:[self.h, self.w, 1]}

        self.request_id = []
        for request in range(requests):
            # self.exec_net.requests[request].set_completion_callback(py_callback=self.callback, py_data=request)
            self.request_id.append(request)


    def start_infer(self,frame,request_ID):
        try :
            frame_in = cv2.resize(frame, (self.w, self.h))
            # frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB).astype(float)
            # frame_in = (frame_in -127)/128
            frame_in = frame_in.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            frame_in = frame_in.reshape((self.n, self.c, self.h, self.w))
            # print(frame_in)

            self.feed_dict[self.input_blob] = frame_in
            self.exec_net.start_async(request_id=request_ID, inputs=self.feed_dict)

        except Exception as e:
            print("Error happen at AG start_infer",e)



    def get_result(self,request_ID):
        self.exec_net.requests[request_ID].wait(-1)
        class_list = self.exec_net.requests[request_ID].output_blobs[self.output_name[0]].buffer[0]
        class_id = np.argmax(class_list)
        result = self.label_map[str(class_id)]
        make = result.split("_")[0]
        model = result.split("_")[1]

        return make, model

    def run(self, frame):
        self.start_infer(frame, 0)
        return self.get_result(0)

class C:
    def __init__(self, model_xml, model_bin, device="CPU", requests=2):
        ie = IECore()
        self.net = ie.read_network(model= model_xml, weights= model_bin)
        self.exec_net = ie.load_network(network=self.net, num_requests=requests, device_name=device)
        self.label_map = {0:"white",1:"gray",2:"yellow",3:"red",4:"green",5:"blue",6:"black"}

        # self.callback = lambda status, py_data : outsource_callback(py_data)


        for blob_name in self.net.input_info:
            if len(self.net.input_info[blob_name].input_data.shape) == 4:
                self.input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(self.net.input_info[blob_name].input_data.shape), blob_name))

        self.output_name = []
        for blob_name in self.net.outputs:
            self.output_name.append(blob_name)

        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape
        self.feed_dict= {self.input_blob:[self.h, self.w, 1]}

        self.request_id = []
        for request in range(requests):
            # self.exec_net.requests[request].set_completion_callback(py_callback=self.callback, py_data=request)
            self.request_id.append(request)


    def start_infer(self,frame,request_ID):
        try :
            frame_in = cv2.resize(frame, (self.w, self.h))
            # frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB).astype(float)
            # frame_in = (frame_in -127)/128
            frame_in = frame_in.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            frame_in = frame_in.reshape((self.n, self.c, self.h, self.w))
            # print(frame_in)

            self.feed_dict[self.input_blob] = frame_in
            self.exec_net.start_async(request_id=request_ID, inputs=self.feed_dict)

        except Exception as e:
            print("Error happen at AG start_infer",e)



    def get_result(self,request_ID):
        self.exec_net.requests[request_ID].wait(-1)
        colour_list = self.exec_net.requests[request_ID].output_blobs[self.output_name[0]].buffer[0]
        colour_ID = np.argmax(colour_list)
        colour = self.label_map[colour_ID]
        return colour

    def run(self, frame):
        self.start_infer(frame, 0)
        return self.get_result(0)

class MMC_service():
    def __init__(self):
        self.MM = MM(model_xml="own_model/VD_MM.xml",
            model_bin="own_model/VD_MM.bin")
        self.C = C(model_xml="intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml",
            model_bin="intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.bin")

    def run(self,frame):
        make, model = self.MM.run(frame.copy())
        colour = self.C.run(frame)
        message = {"make":make,"model":model,"colour":colour}

        return message


