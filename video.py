
from faceswap import faceswap
import cv2

def main():
	# Read image
	image = cv2.imread("image1.jpg")
	# Initialize video capture
	cap = cv2.VideoCapture(0)
	while True:
		# Capture frame
		ret, frame = cap.read()
		faceswap(image, frame)
		# Display frame
		cv2.imshow("Video", frame)
		cv2.imshow("Image", image)
		# Press 'q' to quit video capture
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# Release video capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()