#####
#
# Generates a small subset of data to train with first (so that my laptop doesn't blow up)
#
#####

import cv2 as cv

IMG_PER_CLASS = 50

# Set up webcam
webcam = cv.VideoCapture(0)

print("generate_data - Webcam started")

record_frames = False
folder = "A"


# frame by frame
counter = 0
while True:
	ret, frame = webcam.read()

	cv.imshow("Webcam", frame)

	# Record the frame as data
	if record_frames:
		# Save frame into folder
		print(f"{counter}")
		cv.imwrite(f"../data/raw/{folder}/{folder}{counter}.jpg", frame)
		counter += 1

	# Stop after IMG_PER_CLASS frames
	if counter >= IMG_PER_CLASS:
		counter = 0
		record_frames = False
		# increment letter
		if folder == "Z":
			folder = "A"
		else:
			folder = chr(ord(folder) + 1)

		

	key_code = cv.waitKey(40) & 0xFF

	# when spacebar pressed, record the next IMG_PER_CLASS frames
	if key_code == 32 and not record_frames:
		record_frames = True
		print("Space pressed")

	if key_code == 27: # ESC key to exit
		break


webcam.release()
cv.destroyAllWindows()