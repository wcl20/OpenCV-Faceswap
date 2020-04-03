from imutils import face_utils
import numpy as np
import dlib
import cv2

# Initailize face detector 
detector = dlib.get_frontal_face_detector()
# Initalize facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Define color for facial landmarks
color = (255, 0, 0)

#########################################################################
# OpenCV Drawing Functions
#########################################################################
def draw_triangle(image, triangle, color):
	pt1 = (triangle[0], triangle[1])
	pt2 = (triangle[2], triangle[3])
	pt3 = (triangle[4], triangle[5])
	cv2.line(image, pt1, pt2, color)
	cv2.line(image, pt1, pt3, color)
	cv2.line(image, pt2, pt3, color)

def draw_point(image, point, color):
	px, py = point
	cv2.circle(image, (px, py), 2, color, cv2.FILLED)

def draw_rectangle(image, rect, color):
	x, y, w, h = rect
	cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


#########################################################################
# Face Swap Functions
#########################################################################
def faceswap(source, destinaton):
	"""Swaps a face from source image to destination image.

	Destination image will be modified with face from source image.
	"""

	# Detect faces
	face1 = face_detection(source)
	face2 = face_detection(destinaton)
	if face1 and face2:
		# Get facial landmarks
		landmarks1 = facial_landmarks(source, face1)
		landmarks2 = facial_landmarks(destinaton, face2)
		# Perform Delaunay Triangulation on source image
		triangles = delaunay_triangulation(source, landmarks1)
		if triangles:
			for triangle in triangles:
				# Get landmark composition
				landmark1, landmark2, landmark3 = triangle
				# Get triangle from images
				triangle1 = landmarks1[[landmark1, landmark2, landmark3]]
				triangle2 = landmarks2[[landmark1, landmark2, landmark3]]
				# Create a patch with warped triangle
				patch, mask = warp_triangle(source, triangle1, triangle2)
				# Apply patch to output image
				apply_patch(destinaton, patch, cv2.boundingRect(triangle2), mask)	

def face_detection(image):
	"""Detects faces from an image.

	Returns first face if found, otherwise returns None.
	"""

	# Convert image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Detect faces in frame
	faces = detector(gray, 0)
	# Return face if a face is detected
	return faces[0] if faces else None



def facial_landmarks(image, face, show=False):
	"""Detects facial landmarks from a face.

	Returns an array of facial landmarks of the face.
	"""

	# Convert image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Predict landmarks on face
	landmarks = predictor(gray, face)
	# Convert shape object to numpy array
	landmarks = face_utils.shape_to_np(landmarks)
	if show:
		# Draw landmarks on image
		for landmark in landmarks:
			draw_point(image, landmark, color)
	return landmarks

def delaunay_triangulation(image, landmarks, show=False):
	"""Perform Dealunay Triangulation from an array of landmarks.

	Returns a list of triangles, each triangle represented by its 
	landmark composition.
	"""

	# Create subdiv to include all landmarks
	convexHull = cv2.convexHull(landmarks)
	bounding_box = cv2.boundingRect(convexHull)
	subdiv = cv2.Subdiv2D(bounding_box)
	# For each landmark ...
	for px, py in landmarks:
		# ... Insert landmark
		subdiv.insert((px, py))
	# Perform Delaunay Triangulation
	triangles = subdiv.getTriangleList()
	# Return list of triangles defined by landmark indices
	landmark_triangles = []
	for triangle in triangles:
		# Get points of triangle
		pt1 = (triangle[0], triangle[1])
		pt2 = (triangle[2], triangle[3])
		pt3 = (triangle[4], triangle[5])
		# Find corresponding landmark indices for each point
		index1 = np.where((landmarks == pt1).all(axis=1))[0][0]
		index2 = np.where((landmarks == pt2).all(axis=1))[0][0]
		index3 = np.where((landmarks == pt3).all(axis=1))[0][0]
		# Define a triangle by its landmark composition
		landmark_triangles.append([index1, index2, index3])
	if show:
		# Draw bounding box on image
		draw_rectangle(image, bounding_box, color)
		# Draw triangles on image
		for triangle in triangles:
			draw_triangle(image, triangle, color)
	return landmark_triangles	

def warp_triangle(image, triangle1, triangle2):
	"""Apply affine transformation to a triangular region in image.

	Returns a warped version of the image and a triangular mask the 
	shape of triangle2.
	"""

	# Get bounding rectangle for each triangle
	x1, y1, w1, h1 = cv2.boundingRect(triangle1)
	x2, y2, w2, h2 = cv2.boundingRect(triangle2)
	# Modify triangle coordinates to location of cropped image
	triangle1 = [[triangle1[i][0] - x1, triangle1[i][1] - y1] for i in range(3)]
	triangle2 = [[triangle2[i][0] - x2, triangle2[i][1] - y2] for i in range(3)]
	# Get affine transformation matrix
	M = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
	# Apply transformation to image patch
	patch = image[y1:y1+h1, x1:x1+w1]	
	warp = cv2.warpAffine(patch, M, (w2, h2))
	# Create triangle mask
	mask = np.zeros((h2, w2), dtype=np.uint8)
	cv2.fillConvexPoly(mask, np.int32(triangle2), 255)
	# Mask out triangle region
	warp = cv2.bitwise_and(warp, warp, mask=mask)
	return warp, cv2.bitwise_not(mask)

def apply_patch(image, patch, rect, mask):
	"""Copy a masked patch to an image region.

	Image region specified by rect will be modified by patch.
	"""

	# Find region in image to apply patch
	x, y, w, h = rect 
	region = image[y:y+h, x:x+w]
	# Remove content from image
	region = cv2.bitwise_and(region, region, mask=mask)
	# Apply patch to image
	image[y:y+h, x:x+w] = cv2.add(region, patch)






