from openvino.inference_engine import IECore
import cv2
import numpy as np

class AG:
	def __init__(self, model_xml, model_bin, device="CPU", requests=2):
		ie = IECore()
		self.net = ie.read_network(model= model_xml, weights= model_bin)
		self.exec_net = ie.load_network(network=self.net, num_requests=requests, device_name=device)

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
		age = self.exec_net.requests[request_ID].output_blobs[self.output_name[0]].buffer[0][0][0][0]
		gender_list = self.exec_net.requests[request_ID].output_blobs[self.output_name[1]].buffer[0]
		if np.argmax(gender_list) == 0:
			gender = "female"
		else:
			gender = "male"

		age = round(age*100)


		return age, gender

	def run(self, frame):
		self.start_infer(frame, 0)
		return self.get_result(0)

class E:
	def __init__(self, model_xml, model_bin, device="CPU", requests=2):
		ie = IECore()
		self.net = ie.read_network(model= model_xml, weights= model_bin)
		self.exec_net = ie.load_network(network=self.net, num_requests=requests, device_name=device)

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
		output = self.exec_net.requests[request_ID].output_blobs[self.output_name[0]].buffer[0]
		emotion = np.argmax(output)

		if emotion == 0:
			emotion = "neutral"
		elif emotion == 1:
			emotion = "happy"
		elif emotion == 2:
			emotion = "sad"
		elif emotion == 3:
			emotion = "surprise"
		elif emotion == 4:
			emotion = "anger"

		return emotion

	def run(self, frame):
		self.start_infer(frame, 0)
		return self.get_result(0)

		
class AGE_service():
	def __init__(self):
		self.AG = AG(model_xml="intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml",
			model_bin="intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin")
		self.E = E(model_xml="intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml",
			model_bin="intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin")

	def run(self,frame):
		age, gender = self.AG.run(frame.copy())
		emotion = self.E.run(frame)
		message = {"age":age,"gender":gender,"emotion":emotion}

		return message