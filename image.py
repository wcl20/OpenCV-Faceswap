from faceswap import faceswap
import cv2

def main():
	# Read image
	image1 = cv2.imread("image1.jpg")
	image2 = cv2.imread("image2.jpg")
	# Swap faces 
	faceswap(image1, image2)
	# Display image 
	cv2.imshow("Faceswap", image2)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()