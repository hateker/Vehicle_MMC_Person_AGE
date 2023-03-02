from AGE import AGE_service
from MMC import MMC_service
import cv2
from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()
age_service = AGE_service()
mmc_service = MMC_service()

@app.post("/person_image")
async def person_image(image: UploadFile = File(...)):
	try:
		with open("cache/input_image.jpg", "wb") as buffer:
			shutil.copyfileobj(image.file, buffer)
		frame = cv2.imread("cache/input_image.jpg")
		return age_service.run(frame)
	except Exception as e:
		return e

@app.post("/vehicle_image")
async def vehicle_image(image: UploadFile = File(...)):
	try:
		with open("cache/input_image.jpg", "wb") as buffer:
			shutil.copyfileobj(image.file, buffer)
		frame = cv2.imread("cache/input_image.jpg")
		return mmc_service.run(frame)
	except Exception as e:
		return e