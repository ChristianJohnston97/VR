import csv
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random 

####################################################################
# Christian Johnston
# VR/AR Assignment
# Orientation Tracker via sensor fusion
# NB. Quaternions in form [x, y, z, w] (i.e real part at end)

# Pandas version 0.24.1

# Python 3.7.2


####################################################################
# By considering the data, accelerometerZ is in fact equal to accelerometerY
# the 'up-direction' in the world frame is in fact in the Z direction
colnames = ['time', 'gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'accelerometerX', 'accelerometerZ', 'accelerometerY', 'magnetometerX', 'magnetometerY', 'magnetometerZ']

# read CSV file using pandas
data = pd.read_csv('IMUData.csv', names=colnames)

# selects all the data but the first row (i.e the column names)
data = data.iloc[1:]

# convert the whole dataframe to numeric/float 
data = data.apply(pd.to_numeric)

####################################################################
# returns the row at index i of a given dataframe
def selectRowByIndex(data, i):
	return data.iloc[i]

####################################################################
# just selects the gyroscope columns from data
def selectGyroscopeColumns(data):
	columns = data.iloc[:,1:4]
	return columns

####################################################################
# just selects the accelerometer columns from data
def selectAccelerometerColumns(data):
	columns = data.iloc[:,4:7]
	return columns

####################################################################
# just selects the magnetometer columns from data
def selectMagnetometerColumns(data):
	columns = data.iloc[:,7:10]
	return columns

####################################################################
# gets 1st column of dataset- i.e. the time column 
def getTimingInfo(data):
	timeColumn = data.time.tolist()
	return timeColumn

####################################################################
# function to return radians to degree
def radiansToDegrees(radian):
	return math.degrees(radian)

####################################################################
# function to return radians to degree
def degreesToRadians(degree):
	return math.radians(degree)

# apply the above (converting rotational rate from degrees to radians/s)
data['gyroscopeX'] = data['gyroscopeX'].apply(degreesToRadians)
data['gyroscopeY'] = data['gyroscopeY'].apply(degreesToRadians)
data['gyroscopeZ'] = data['gyroscopeZ'].apply(degreesToRadians)

####################################################################
# function to calculate the size/norm of the 3 components
def calculateNorm(x, y, z):
	norm = math.sqrt(x**2 + y**2 + z**2)
	return norm

####################################################################
# function to normalise magnitude of vector
def normaliseVector(vector):
	[x, y, z] = vector
	norm = calculateNorm(x, y, z)
	# test for zero vector
	if norm == 0:
		newVector = [0, 0, 0]
	else:
		newVector = [val / norm for val in vector]
	return newVector

####################################################################

acceleromterDF = selectAccelerometerColumns(data)
magnetometerDF = selectMagnetometerColumns(data)

# save the un-normalised versions for later plotting
unNormalisedAcceleromterData = acceleromterDF
unNormalisedMagnetometerData = magnetometerDF

# create copies
acceleromterDF = acceleromterDF.copy()
magnetometerDF = magnetometerDF.copy()

# calculate accelerometer norm 
acceleromterDF['acceleromterNorm'] = acceleromterDF.apply(lambda row : calculateNorm(row['accelerometerX'], row['accelerometerY'], row['accelerometerZ']), axis = 1)

# calculate magnetometer norm 
magnetometerDF['magnetometerNorm'] = magnetometerDF.apply(lambda row : calculateNorm(row['magnetometerX'], row['magnetometerY'], row['magnetometerZ']), axis = 1)

# normalise accelerometer data (divide by norm)
# this takes special care as a division by zero would produce an inf or a NaN and 
# (+-inf or NaN ) are set to the zero vector.
data['accelerometerX'] = data['accelerometerX']/acceleromterDF['acceleromterNorm'].fillna(0).replace([np.inf, -np.inf], 0)
data['accelerometerY'] = data['accelerometerY']/acceleromterDF['acceleromterNorm'].fillna(0).replace([np.inf, -np.inf], 0)
data['accelerometerZ'] = data['accelerometerZ']/acceleromterDF['acceleromterNorm'].fillna(0).replace([np.inf, -np.inf], 0)

# normalise magnetometer data (divide by norm)
# this takes special care as a division by zero would produce an inf or a NaN and 
# (+-inf or NaN ) are set to the zero vector.
data['magnetometerX'] = data['magnetometerX']/magnetometerDF['magnetometerNorm'].fillna(0).replace([np.inf, -np.inf], 0)
data['magnetometerY'] = data['magnetometerY']/magnetometerDF['magnetometerNorm'].fillna(0).replace([np.inf, -np.inf], 0)
data['magnetometerZ'] = data['magnetometerZ']/magnetometerDF['magnetometerNorm'].fillna(0).replace([np.inf, -np.inf], 0)

# get the time column
timeColumn = getTimingInfo(data)

###################################################################

### QUESTION 1 ###

###################################################################
# Convert Euler angle readings (radians) to quaternions 
# For order: roll pitch yaw (X Y Z)
def euler_to_quaternion_XYZ(X, Y, Z):
	c1 = math.cos(X/2)
	s1 = math.sin(X/2)
	c2 = math.cos(Y/2)
	s2 = math.sin(Y/2)
	c3 = math.cos(Z/2)
	s3 = math.sin(Z/2)

	x = s1*c2*c3 + c1*s2*s3
	y = c1*s2*c3 - s1*c2*s3
	z = c1*c2*s3 + s1*s2*c3
	w = c1*c2*c3 - s1*s2*s3

	quaternion = [x,y,z,w]
	return quaternion	

###################################################################
# Returns euler angles (roll, pitch, yaw) (X Y Z)
def quaternion_to_euler_XYZ(quaternion):
	qx, qy, qz, qw = quaternion
	if ((qx*qz + qw*qy) > 0.499):
		eulerX = 0
		eulerY = 2 * math.atan2(qx,qw)
		eulerZ = math.pi/2
	elif ((qx*qz + qw*qy) < -0.499):
		eulerX = 0
		eulerY = -2 * math.atan2(qx,qw)
		eulerZ = - math.pi/2
	else:
		eulerX = -math.atan2(2*qy*qz - 2*qw*qx , 2*(qw**2) + 2*(qz**2) - 1)
		eulerY = math.asin (2*(qx*qz + qw*qy));
		eulerZ = -math.atan2(2*qx*qy - 2*qw*qz , 2*(qw**2) + 2*(qx**2) - 1)

	return eulerX, eulerY, eulerZ

###################################################################
# convert a quaternion to its conjugate (inverse rotation)
def quaternionConjugate(quaternion):
	[qx, qy, qz, qw] = quaternion
	conjugate = [-qx, -qy, -qz, qw]
	return conjugate

####################################################################
#calculate the quaternion product of quaternion a and b 
def quaternionProduct(quaternion1, quaternion2):
	[x0, y0, z0, w0] = quaternion1
	[x1, y1, z1, w1] = quaternion2
	x = (w1*x0) + (w0*x1) - (y1*z0) + (y0*z1)
	y = (w1*y0) + (x1*z0) + (y1*w0) - (z1*x0)
	z = (w1*z0) - (x1*y0) + (y1*x0) + (z1*w0)
	w = (w0*w1) - (x0*x1) - (y0*y1) - (z0*z1)
	product = [x,y,z,w]
	return product

####################################################################
# Turn axis angle to quaterion, angles in radians 
def axisAngleToQuaternion(axis, angle):
	axis = normaliseVector(axis)
	[ax, ay, az] = axis
	qw = math.cos(angle/2)
	qx = ax * math.sin(angle/2)
	qy = ay * math.sin(angle/2)
	qz = az * math.sin(angle/2)
	quaternion = [qx, qy, qz, qw]
	return quaternion

####################################################################
# function to turn axis angle to quaterion, angles must be in radians
def quaternionToAxisAngle(quaternion):
	[qx, qy, qz, qw] = quaternion
	angle = 2*math.acos(qw)
	x = qx / math.sqrt(1-qw*qw)
	y = qy / math.sqrt(1-qw*qw)
	z = qz / math.sqrt(1-qw*qw)
	axis = [x, y, z]
	return axis, angle

####################################################################

### QUESTION 2 ### 		Dead reckoning filter

####################################################################
# q'[k+1] = q'[k] * q(v, angle)
# calculate current position (at stage k) by using a previously determined position 
# (starting at the identity quaternion [0,0,0,1]) and advancing that position based upon 
# an estimated angular speed over the elapsed time
# iteratively estimate orientation using gyroscope (anglular velocity)
# In short: integrate grysocope readings- get orientation
def deadReckoningFilter(data):

	# estimated orientation list
	estimatedOrientationList = np.zeros((len(data), 4))

	# get accelerometer columns
	gyroscopeData = selectGyroscopeColumns(data)

	# initial orientation is the identity quaternion (w = 1)
	estimatedOrientationList[0] = [0,0,0,1]

	# get the time column 
	timing = getTimingInfo(data)

	# start at k = 1
	k = 1

	# loop through all of the stages 
	while k < len(gyroscopeData):

		# selct the row at stage k
		row = selectRowByIndex(gyroscopeData, k)

		# get the data
		gyroX = row.gyroscopeX
		gyroY = row.gyroscopeY
		gyroZ = row.gyroscopeZ

		# angular velocity at time k
		angularVelocity = [gyroX, gyroY, gyroZ]

		# calculate the norm (rate of rotation about the axis below)
		norm = calculateNorm(gyroX, gyroY, gyroZ)

		# Axis of rotation (i.e. normalise the angular velocity)
		axis = normaliseVector(angularVelocity)

		# change in time
		changeInTime = timing[k] - timing[k-1]

		# Amount of rotation during time delta t
		angle = norm * changeInTime

		# Orientation change over time
		changeInOrientation = axisAngleToQuaternion(axis, angle)

		# estimated new orientation
		newOrientation = quaternionProduct(estimatedOrientationList[k-1], changeInOrientation)

		# append to list
		estimatedOrientationList[k] = newOrientation

		# move onto next stage
		k += 1

	return estimatedOrientationList

####################################################################

### QUESTION 3 #### gravity-based tilt correct using the accelerometer

####################################################################

# Instead of just integrating the gyroscope, include accelerometer information too

####################################################################
# Transform acceleration measurements into the global frame 
# a^ = q * a * (q)^-1 
# a = acceleration in local frame, a^ = acceleration in global frame
# q is the previous drift-corrected orientation updated with the latest gyroscope reading
def accelerationToGlobalFrame(localAcceleration, q):

	# compute inverse / conjugate
	q_inverse = quaternionConjugate(q)

	# apply double quaternion product 
	intermediaryResult = quaternionProduct(q, localAcceleration)

	# Multiplication of quaternions is associative
	globalAcceleration = quaternionProduct(intermediaryResult, q_inverse)

	# this is an estimation of the up_vector (as a quaternion)
	return globalAcceleration

####################################################################
# Vector helper functions

# normalise a vector (different form to previous normalisation funtion)
def unit_vector(x, y, z):
	vector = [x, y, z]
	vector = vector / np.linalg.norm(vector)
	[newX, newY, newZ] = vector
	return newX, newY, newZ

# dot product between two vectors
def dotproduct(v1, v2):
	return sum((a*b) for a, b in zip(v1, v2))

# length of a vector
def length(vector):
	[x, y, z] = vector
	length = math.sqrt(x**2 + y**2 + z**2)
	return length

# angle in radians between two vectors
def angle(v1, v2):
	v1 = normaliseVector(v1)
	v2 = normaliseVector(v2)
	angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
	return angle

####################################################################
# To calculate the axis, project the global acceleration onto the XZ plane
# the tilt axis is orthogonal: t = (az, 0, -ax)
def calculateTiltAxis(acceleration):
	[ax, ay, az] =  acceleration
	tilt_axis = [az, 0, -ax]
	return tilt_axis

####################################################################
# The tilt error is the angle between the vector obtained from the accelerometer (in globlal frame)
# and the up vector (0, 1, 0)
# globalAcc is an estimate of up-vector 
def calcuateTiltError(globalAcc):
	up_vector = [0, 1, 0]
	tilt_error = angle(globalAcc, up_vector)
	return tilt_error

####################################################################
# Maintains a running average of local acceleromter values 
# from past 10 readings (or as many as possible up to 10)
def calcuateRunningAverage(xAccQueue, yAccQueue, zAccQueue, accX, accY, accZ):
	if len(xAccQueue) < 10:
		xAccQueue.insert(0, accX)
		yAccQueue.insert(0, accY)
		zAccQueue.insert(0, accZ)
	else:
		xAccQueue.pop()
		yAccQueue.pop()
		zAccQueue.pop()
		xAccQueue.insert(0, accX)
		yAccQueue.insert(0, accY)
		zAccQueue.insert(0, accZ)

	avgX = sum(xAccQueue) / len(xAccQueue)
	avgY = sum(yAccQueue) / len(yAccQueue)
	avgZ = sum(zAccQueue) / len(zAccQueue)

	return xAccQueue, yAccQueue, zAccQueue, avgX, avgY, avgZ

####################################################################
# Correct orientation by reducing tilt error 
# tilt error = component of all drift error except in the vertical y plane
# Complementary filter to fuse the two signals: (gyroscopic data and accelerometer data)
# a << 1 (eg. 0.01, 0.1)
# returns orientation with drift error removed
def tiltCorrection(orientationEstimateList, alpha):

	# for accelerometer running average
	xAccQueue = []
	yAccQueue = []
	zAccQueue = []

	# list of new estimated orientation (with drift error removed)
	tiltCorrected = np.zeros((len(data), 4))

	# accelerometer data
	accelerometerData = selectAccelerometerColumns(data)

	# gyroscope data
	gyroscopeData = selectGyroscopeColumns(data)

	# initial drift corrected orientation: identity quaternion
	tiltCorrected[0] = [0, 0, 0, 1]

	k = 1

	while k < len(orientationEstimateList):

		# get the latest gyroscope reading
		row = selectRowByIndex(gyroscopeData, k)

		# get the data
		gyroX = row.gyroscopeX
		gyroY = row.gyroscopeY
		gyroZ = row.gyroscopeZ

		# convert to quaternion
		gyroQuaternion = euler_to_quaternion_XYZ(gyroX, gyroY, gyroZ)

		# quaternion multiplication of best orientation estimate from stage k (gyroscope + accelerometer)
		# with the latest gyroscope reading
		# returns newest orientation estimate (with drift error) - needs correcting
		q = quaternionProduct(tiltCorrected[k-1], gyroQuaternion)

		# get the latest acceleromter reading
		acc = selectRowByIndex(accelerometerData, k)
		accX = acc.accelerometerX
		accY = acc.accelerometerY
		accZ = acc.accelerometerZ

		# calculates a running average from the last 10 accelerometer values
		xAccQueue, yAccQueue, zAccQueue, accAvgX, accAvgY, accAvgZ = calcuateRunningAverage(xAccQueue, yAccQueue, zAccQueue, accX, accY, accZ)

		# average acceleration vector to quaternion (set w=0)
		localAccelerationQuat = [accAvgX, accAvgY, accAvgZ, 0]

		# estimate global acceleration (quaternion)
		globalAccelerationQuat = accelerationToGlobalFrame(localAccelerationQuat, q)

		# global acceleration UNIT vector (take the x,y,z from quaternion)
		globalAccelerationVector = globalAccelerationQuat[:3]

		# Calculate tilt axis 
		tilt_axis = calculateTiltAxis(globalAccelerationVector)

		# Calculate tilt error (from up estimate, i.e. global acceleration)
		tilt_error = calcuateTiltError(globalAccelerationVector)

		# orientation correction via complementary filter (equation 10 in LaValle's)
		quaternion = axisAngleToQuaternion(tilt_axis, -alpha*tilt_error)

		# new orientation to multiply with previous orientation estimate (from gyro integration)
		correctedOrientation = quaternionProduct(quaternion, orientationEstimateList[k])

		# append to list
		tiltCorrected[k] = correctedOrientation

		# increment k
		k += 1

	return tiltCorrected

####################################################################

### QUESTION 4 #### Magnetometer based yaw correction 

####################################################################

# Use magnetometer to estimate if there has been yaw drift by comparing a 
# reference magnetometer reading taken at the beginning of the IMU rotations 
# and each current magnetometer reading 

####################################################################
# magnetometer value before drift accumulates
def getMagnetometerRefValues(magnetometerData):
	row = selectRowByIndex(magnetometerData, 0)
	magRefX = row.magnetometerX
	magRefY = row.magnetometerY
	magRefZ = row.magnetometerZ
	magRefValues = [magRefX, magRefY, magRefZ]
	return magRefValues

####################################################################
# Transform magnetometer readings into the estimated global frame
def magnetometerToGlobal(orientationEstimate, magnetometer):
	# find inverse/conjugate of orientation
	orientationInverse = quaternionConjugate(orientationEstimate)
	# double quaternion product 
	intermediaryResult = quaternionProduct(orientationInverse, magnetometer)
	final = quaternionProduct(intermediaryResult, orientationEstimate)
	return final

####################################################################
# average magnetometer values 
def averageMagnetometerValues(xMagQueue, yMagQueue, zMagQueue, magX, magY, magZ):
	if len(xMagQueue) < 10:
		xMagQueue.insert(0, magX)
		yMagQueue.insert(0, magY)
		zMagQueue.insert(0, magZ)
	else:
		xMagQueue.pop()
		yMagQueue.pop()
		zMagQueue.pop()
		xMagQueue.insert(0, magX)
		yMagQueue.insert(0, magY)
		zMagQueue.insert(0, magZ)

	avgX = sum(xMagQueue) / len(xMagQueue)
	avgY = sum(yMagQueue) / len(yMagQueue)
	avgZ = sum(zMagQueue) / len(zMagQueue)

	return xMagQueue, yMagQueue, zMagQueue, avgX, avgY, avgZ

####################################################################
# Correction is a rotation about the y-axis only 
# Yaw detection complementary filter
# Use of magnetometer to estimate if there has been yaw drift
# Correct this via complementary filter with paramater alpha
def yawCorrection(orientationEstimateList, alpha):

	# list of new estimated orientation (with yaw error removed)
	newEstimatedOrientation = np.zeros((len(data), 4))

	# accelerometer data
	magnetometerData = selectMagnetometerColumns(data)

	# gyroscope data 
	gyroscopeData = selectGyroscopeColumns(data)

	# get the reference magnetometer values
	[magRefX, magRefY, magRefZ] = getMagnetometerRefValues(magnetometerData)

	# convert magnetometer reference values (vector) to quaternion
	magRefValuesQuat = [magRefX, magRefY, magRefZ, 0]

	# initial orientation (identity quaternion)
	newEstimatedOrientation[0] = [0, 0, 0, 1]

	xMagQueue = []
	yMagQueue = []
	zMagQueue = []

	# convert reference magnetometer values to global frame
	globalMagRefQuat = magnetometerToGlobal([0, 0, 0, 1], magRefValuesQuat)

	k = 1

	while k < len(orientationEstimateList):

		# get the latest gyroscope reading
		row = selectRowByIndex(gyroscopeData, k)

		# get the data
		gyroX = row.gyroscopeX
		gyroY = row.gyroscopeY
		gyroZ = row.gyroscopeZ

		# convert to quaternion
		gyroQuaternion = euler_to_quaternion_XYZ(gyroX, gyroY, gyroZ)

		# quaternion multiplication of best orientation estimate from stage k 
		# (gyroscope + accelerometer + magnetometer) with the latest gyroscope reading
		# returns newest orientation estimate (with yaw error) - needs correcting
		newOrientation = quaternionProduct(newEstimatedOrientation[k-1], gyroQuaternion)

		# get the specific row
		magXYZ = selectRowByIndex(magnetometerData, k)
		magX = magXYZ.magnetometerX
		magY = magXYZ.magnetometerY
		magZ = magXYZ.magnetometerZ

		# calculate rolling average of magnetomter values 
		xMagQueue, yMagQueue, zMagQueue, accMagX, accMagY, accMagZ = averageMagnetometerValues(xMagQueue, yMagQueue, zMagQueue, magX, magY, magZ)

		# convert magnetometer vector to quaternion
		magQuaternion = [accMagX, accMagY, accMagZ, 0]

		# map each acceleration from local frame to global frame
		globalMagQuat = magnetometerToGlobal(newOrientation, magQuaternion)

		# Project magnetometer values onto the horizontal XZ plane
		theta = math.atan2(globalMagQuat[0], globalMagQuat[2])

		# get the x and z value of magnetometer refererence values 
		thetaRef = math.atan2(globalMagRefQuat[0], globalMagRefQuat[2])

		# calculate angular difference / error 
		angle = theta - thetaRef

		# axis angle to quaternion
		quaternion = axisAngleToQuaternion([0,1,0], -alpha*(angle))

		# orientation estimate (from gyroscope + accelerometer) as a quaternion
		orientation = quaternionProduct(quaternion, orientationEstimateList[k])

		# append to list
		newEstimatedOrientation[k] = orientation

		# increment k
		k += 1

	return newEstimatedOrientation

####################################################################
#### QUESTION 5 #### PLOTS

####################################################################
# Plots all three plots for input dataset
# pass in the un-normalised data as parmaters
def triple2DPlot1(data, accData, magData):
	plt.figure(figsize=(14,6))
	plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.96, wspace=0.20, hspace=0.58)

	plt.subplot(3, 1, 1)
	plt.title('Tri-axial angular rate as a function of time', fontsize=10)

	# Read in the gyroscope data and turn into list 
	gyroscopeX = data.gyroscopeX.tolist()
	gyroscopeY = data.gyroscopeY.tolist()
	gyroscopeZ = data.gyroscopeZ.tolist()
	gyroscopeX = list(map(radiansToDegrees, gyroscopeX))
	gyroscopeY = list(map(radiansToDegrees, gyroscopeY))
	gyroscopeZ = list(map(radiansToDegrees, gyroscopeZ))
	roll, = plt.plot(timeColumn, gyroscopeX, 'r', label='Roll')
	pitch, = plt.plot(timeColumn, gyroscopeY, 'g', label='Pitch')
	yaw, = plt.plot(timeColumn, gyroscopeZ, 'b', label='Yaw')
	plt.legend(handles=[roll, pitch, yaw], loc = 3)
	plt.xlabel('Time (s)', fontsize=10)
	plt.ylabel('Angular Rate (deg/s)', fontsize=10)

	plt.subplot(3, 1, 2)
	plt.title('Tri-axial acceleration rate as a function of time', fontsize=10)
	accelerometerX = accData.accelerometerX.tolist()
	accelerometerY = accData.accelerometerY.tolist()
	accelerometerZ = accData.accelerometerZ.tolist()
	X, = plt.plot(timeColumn, accelerometerX, 'r', label='X')
	Y, = plt.plot(timeColumn, accelerometerY, 'g', label='Y')
	Z, = plt.plot(timeColumn, accelerometerZ, 'b', label='Z')
	plt.legend(handles=[X, Y, Z], loc = 3)
	plt.xlabel('Time (s)', fontsize=10)
	plt.ylabel('Acceleration (m/s2)', fontsize=10)

	plt.subplot(3, 1, 3)
	plt.title('Tri-axial magnetic flux as a function of time', fontsize=10)
	magnetometerX = magData.magnetometerX.tolist()
	magnetometerY = magData.magnetometerY.tolist()
	magnetometerZ = magData.magnetometerZ.tolist()
	X, = plt.plot(timeColumn, magnetometerX, 'r', label='X')
	Y, = plt.plot(timeColumn, magnetometerY, 'g', label='Y')
	Z, = plt.plot(timeColumn, magnetometerZ, 'b', label='Z')

	plt.legend(handles=[X, Y, Z], loc = 3)
	plt.xlabel('Time (s)', fontsize=10)
	plt.ylabel('Magnetic Flux (G)', fontsize=10)

	plt.show()

####################################################################
# Individual Plots

####################################################################
# tri-axial angular rate in deg/s as a function of time
def gyroscopePlot(data):
	# Read in the gyroscope data and turn into list 
	gyroscopeX = data.gyroscopeX.tolist()
	gyroscopeY = data.gyroscopeY.tolist()
	gyroscopeZ = data.gyroscopeZ.tolist()

	gyroscopeX = map(radiansToDegrees, gyroscopeX)
	gyroscopeY = map(radiansToDegrees, gyroscopeY)
	gyroscopeZ = map(radiansToDegrees, gyroscopeZ)

	roll, = plt.plot(timeColumn, gyroscopeX, 'r', label='Roll')
	pitch, = plt.plot(timeColumn, gyroscopeY, 'g', label='Pitch')
	yaw, = plt.plot(timeColumn, gyroscopeZ, 'b', label='Yaw')

	plt.legend(handles=[roll, pitch, yaw])

	plt.xlabel('Time (s)')
	plt.ylabel('Angular Rate (deg/s)')
	plt.show()

####################################################################
# tri-axial acceleration (m/s2) as a function of time
def accelerometerPlot(data):
	# Read in the accelerometer data and turn into list 
	accelerometerX = data.accelerometerX.tolist()
	accelerometerY = data.accelerometerY.tolist()
	accelerometerZ = data.accelerometerZ.tolist()

	X, = plt.plot(timeColumn, accelerometerX, 'r', label='X')
	Y, = plt.plot(timeColumn, accelerometerY, 'g', label='Y')
	Z, = plt.plot(timeColumn, accelerometerZ, 'b', label='Z')

	plt.legend(handles=[X, Y, Z])

	plt.xlabel('Time (s)')
	plt.ylabel('Acceleration (m/s2)')
	plt.show()

####################################################################
#tri-axial magnetic flux in Gauss (G) as a function of time 
def magnetometerPlot(data):
	# Read in the magnetometer data and turn into list 
	magnetometerX = data.magnetometerX.tolist()
	magnetometerY = data.magnetometerY.tolist()
	magnetometerZ = data.magnetometerZ.tolist()

	X, = plt.plot(timeColumn, magnetometerX, 'ro', label='X')
	Y, = plt.plot(timeColumn, magnetometerY, 'go', label='Y')
	Z, = plt.plot(timeColumn, magnetometerZ, 'bo', label='Z')

	plt.legend(handles=[X, Y, Z])

	plt.xlabel('Time (s)')
	plt.ylabel('Magnetic Flux (G)')
	plt.show()

####################################################################
# Generate individual 2D plot for orientation estimates
def plot_2d(orientationList):
	yaw_list = []
	pitch_list = []
	roll_list = []

	# for each quaternion in the list 
	for quaternion in orientationList:
		[qx, qy, qz, qw] = quaternion

		yaw_list.append(qz)
		pitch_list.append(qy)
		roll_list.append(qx)

		# convert back to Euler angles
		roll, pitch, yaw = quaternion_to_euler_XYZ(quaternion)

		yaw_list.append(radiansToDegrees(yaw))
		pitch_list.append(radiansToDegrees(pitch))
		roll_list.append(radiansToDegrees(roll))

	plt.figure(figsize=(14,6))
	yaw, = plt.plot(timeColumn, yaw_list, 'r', label='Yaw')
	pitch, = plt.plot(timeColumn, pitch_list, 'g', label='Pitch')
	roll, = plt.plot(timeColumn, roll_list, 'b', label='Roll')

	plt.legend(handles=[roll, pitch, yaw])

	plt.xlabel('Time (s)')
	plt.ylabel('Tri-axial Euler Angles (degrees)')
	plt.show()

####################################################################
# Plots all 3 estimates on one graph
# If degrees = true, converts quaternions to euler angles in radians
# and then from radians to degrees before plotting
def triple2DPlot2(orientation1, orientation2, orientation3, degrees=True):

	plt.figure(figsize=(14,6))
	plt.subplot(3, 1, 1)
	plt.title('Gyroscope Integration', fontsize=8)
	plt.xlabel('Time (s)', fontsize=8)
	plt.ylabel('Tri-axial Euler Angles (degrees)', fontsize=8)


	yaw_list = []
	pitch_list = []
	roll_list = []

	# for each quaternion in the list 
	for quaternion in orientation1:
		[qx, qy, qz, qw] = quaternion

		if degrees == True:
			roll, pitch, yaw = quaternion_to_euler_XYZ(quaternion)
			yaw_list.append(radiansToDegrees(yaw))
			pitch_list.append(radiansToDegrees(pitch))
			roll_list.append(radiansToDegrees(roll))
		else:
			yaw_list.append(qz)
			pitch_list.append(qy)
			roll_list.append(qx)

	yaw, = plt.plot(timeColumn, yaw_list, 'r', label='Yaw')
	pitch, = plt.plot(timeColumn, pitch_list, 'g', label='Pitch')
	roll, = plt.plot(timeColumn, roll_list, 'b', label='Roll')
	plt.legend(handles=[roll, pitch, yaw], loc = 3)

	
	plt.subplot(3, 1, 2)
	plt.subplots_adjust(left=0.07, bottom=0.08, right=0.98, top=0.95, wspace=0.20, hspace=0.47)
	plt.title('Gyroscope Integration with Accelerometer Drift Correction', fontsize=8)
	plt.xlabel('Time (s)', fontsize=8)
	plt.ylabel('Tri-axial Euler Angles (degrees)', fontsize=8)

	yaw_list = []
	pitch_list = []
	roll_list = []

	# for each quaternion in the list 
	for quaternion in orientation2:
		[qx, qy, qz, qw] = quaternion

		if degrees == True:
			roll, pitch, yaw = quaternion_to_euler_XYZ(quaternion)
			yaw_list.append(radiansToDegrees(yaw))
			pitch_list.append(radiansToDegrees(pitch))
			roll_list.append(radiansToDegrees(roll))
		else:
			yaw_list.append(qz)
			pitch_list.append(qy)
			roll_list.append(qx)

	yaw, = plt.plot(timeColumn, yaw_list, 'r', label='Yaw')
	pitch, = plt.plot(timeColumn, pitch_list, 'g', label='Pitch')
	roll, = plt.plot(timeColumn, roll_list, 'b', label='Roll')
	plt.legend(handles=[roll, pitch, yaw], loc = 3)

	
	plt.subplot(3, 1, 3)
	plt.title('Gyroscope Integration with Accelerometer Drift Correction and Magnetometer Yaw Correction', fontsize=8)
	plt.xlabel('Time (s)',  fontsize=8)
	plt.ylabel('Tri-axial Euler Angles (degrees)', fontsize=8)

	yaw_list = []
	pitch_list = []
	roll_list = []

	# for each quaternion in the list 
	for quaternion in orientation3:
		[qx, qy, qz, qw] = quaternion

		if degrees == True:
			roll, pitch, yaw = quaternion_to_euler_XYZ(quaternion)
			yaw_list.append(radiansToDegrees(yaw))
			pitch_list.append(radiansToDegrees(pitch))
			roll_list.append(radiansToDegrees(roll))
		else:
			yaw_list.append(qz)
			pitch_list.append(qy)
			roll_list.append(qx)

	yaw, = plt.plot(timeColumn, yaw_list, 'r', label='Yaw')
	pitch, = plt.plot(timeColumn, pitch_list, 'g', label='Pitch')
	roll, = plt.plot(timeColumn, roll_list, 'b', label='Roll')
	plt.legend(handles=[roll, pitch, yaw], loc = 3)

	plt.show()	

####################################################################
# Q5 - 3D animation
# create three side-by-side 3D animated plots that show 3 perpendicular vectors (XYZ) in space
# corresponding to the 3 main axis of your IMU that rotate in time as the IMU is being rotated
# i.e. one for each method (gyro, gyro+acc, gyro+acc+mag)

####################################################################
# Quaternion Vector Multiplication
# Rotating a vector by a quaternion
def quaternionVectorRotation(vector, orientation):
	orientationConjugate = quaternionConjugate(orientation)
	intermediaryResult = quaternionProduct(orientation, vector)
	finalResult = quaternionProduct(intermediaryResult, orientationConjugate)
	return finalResult

####################################################################
# update the vectors
def dataGen(k, orientationList, lines):

	# get the orientation (as a quaternion) at stage k 
	orientation = orientationList[k]

	# Quaternion product of the unit vector representing an axis with the current orientation
	# This gives the DIRECTION of the axis at that timestep.
	# x,y,z components of the quaternion are a UNIT vector for the direction of the axis
	[x1, y1, z1, _] = quaternionVectorRotation([1,0,0,0], orientation)
	[x2, y2, z2, _] = quaternionVectorRotation([0,1,0,0], orientation)
	[x3, y3, z3, _] = quaternionVectorRotation([0,0,1,0], orientation)

	cartesianData = [[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]]

	# draw lines to point from origin
	# set_data sets X and Y positions, set_3d_properties sets z-values
	for line, data in zip(lines, cartesianData):
		line.set_data([0, data[0]], [0, data[1]])
		line.set_3d_properties([0, data[2]])

	return lines

####################################################################
# 3D orientation plot for question 5
# individual plots
# 256 frames / s 
# (1/256)*1000 = 3.90625
def create3DOrientationPlot(orientationData, sampleFrac):
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Orientation 3D Animation')

	data_length = int(len(orientationData)*sampleFrac)
	orientationDataSampled = [orientationData[i] for i in sorted(random.sample(range(len(orientationData)), data_length))]

	# data = original orthogonal unit vector axis 
	data = [[[0,1], [0,0], [0,0]], [[0,0], [0,1], [0,0]], [[0,0], [0,0], [0,1]]]
	lines = [ax.plot(dat[0], dat[1], dat[2])[0] for dat in data]
	line_animation = animation.FuncAnimation(fig, dataGen, fargs=(orientationDataSampled, lines), interval = 3.90625, blit = True, repeat=False)
	plt.show()

####################################################################
# Q5 multiple 3D side by side plots 
# this uses the function dataGen above to update the 3D plot
def sideBySide3D(orientation1, orientation2, orientation3, actualSpeed):
	fig = plt.figure(figsize=(14,6))
	plt.subplots_adjust(left=0.10, bottom=0.11, right=0.94, top=0.88, wspace=0.20, hspace=0.20)

	ax = plt.subplot(1, 3, 1, projection = "3d")
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Gyroscope Integration Orientation 3D Animation')

	# data = original orthogonal unit vector axis 
	data = [[[0,1], [0,0], [0,0]], [[0,0], [0,1], [0,0]], [[0,0], [0,0], [0,1]]]
	lines1 = [ax.plot(dat[0], dat[1], dat[2])[0] for dat in data]
	line_animation1 = animation.FuncAnimation(fig, dataGen, fargs=(orientation1, lines1), interval = 1, blit = True)

	ax = plt.subplot(1, 3, 2, projection = "3d")
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Tilt Corrected Orientation 3D Animation')
	lines2 = [ax.plot(dat[0], dat[1], dat[2])[0] for dat in data]
	line_animation2 = animation.FuncAnimation(fig, dataGen, fargs=(orientation2, lines2), interval = 1, blit = True)

	ax = plt.subplot(1, 3, 3, projection = "3d")
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Yaw Corrected Orientation 3D Animation')
	lines3 = [ax.plot(dat[0], dat[1], dat[2])[0] for dat in data]
	line_animation3 = animation.FuncAnimation(fig, dataGen, fargs=(orientation3, lines3), interval = 1, blit = True)

	plt.show()

####################################################################
#### QUESTION 6 #### Positional tracking
# kinematic head model 
# Double integrate the acceleromter

# The coordinate frame origin for positional tracking is located at the midpoint between 
# the retinas --> retinal center frame (RCF)

# Distance from RCF to the base of the neck is l2
# Length of torso is l1
# p = [q1 * (0,l1,0) * (q1)^-1] + [q * (0,l2,0) * (q)^-1]
# p = r1 + r (two terms in equation above)

####################################################################
# As per Eq13 in LaValles paper 
def headModelMapping(orientationEstimate, length):

	# turn a vector into a quaternion 
	quaternion = [0, length, 0, 0]

	# compute inverse / conjugate
	orientationInverse = quaternionConjugate(orientationEstimate)

	# apply double quaternion product 
	intermediaryResult = quaternionProduct(orientationEstimate, quaternion)

	# Multiplication of quaternions is associative
	finalPosition = quaternionProduct(intermediaryResult, orientationInverse)

	return finalPosition

####################################################################
# Double integrate acceleration to get position estimate  
def accelerationToPosition(data):

	# get accelerometer data
	accelerometerData = selectAccelerometerColumns(data)

	# estimated position list
	positionList = np.zeros((len(accelerometerData), 4))

	# velocity list
	velocityList = np.zeros((len(accelerometerData), 4))

	# headModelMapping list
	headModelMappingList = np.zeros((len(accelerometerData), 4))

	# initial velocity is the identity quaternion 
	velocityList[0] = [0,0,0,1]

	# initial position is the identity quaternion 
	positionList[0] = [0,0,0,1]

	# get the time column 
	timing = getTimingInfo(data)

	k = 1

	# loop through all of the stages 
	while k < len(accelerometerData):

		# selct the accelerometer row at stage k
		row = selectRowByIndex(accelerometerData, k)

		# get the data
		accX = row.accelerometerX
		accY = row.accelerometerY
		accZ = row.accelerometerZ

		# acceleration at time k 
		acceleration = [accX, accY, accZ]

		# calculate the norm (overall rate of acceleration)
		norm = calculateNorm(accX, accY, accZ)

		# Axis of acceleration (i.e. normalise the acceleration)
		axis = normaliseVector(acceleration)

		# change in time
		changeInTime = timing[k] - timing[k-1]

		# Amount of acceleration during time delta t
		angle = norm * changeInTime
		
		# velocity change over time
		changeInVelocity = axisAngleToQuaternion(axis, angle)

		# estimated new velocity (as a quaternion)
		newVelocityQuat = quaternionProduct(changeInVelocity, velocityList[k-1])

		# add to list
		velocityList[k] = newVelocityQuat

		# new velocity as a vector (take first 3 values)
		newVelocityVector = newVelocityQuat[:3]

		# calculate the norm (overall rate of velocity)
		norm2 = calculateNorm(newVelocityVector[0], newVelocityVector[1], newVelocityVector[2])

		# Axis of velocity (i.e. normalise the velocity)
		axis2 = normaliseVector(newVelocityVector)

		# Amount of velocity during time delta t
		angle2 = norm2 * changeInTime
			
		# position change over time
		changeInPosition = axisAngleToQuaternion(axis2, angle2)

		# estimated new position
		newPosition = quaternionProduct(changeInPosition, positionList[k-1])

		# add to list 
		positionList[k] = newPosition

		k+=1
		
	return positionList

####################################################################
# integration of linear acceleration (in global frame of head)
def accelerationToLinearVelocity(data, orientation):

	# get accelerometer data
	accelerometerData = selectAccelerometerColumns(data)

	# velocity list
	velocityList = np.zeros((len(accelerometerData), 4))

	# initial velocity is the identity quaternion 
	velocityList[0] = [0,0,0,1]

	# get the time column 
	timing = getTimingInfo(data)

	k = 1

	# loop through all of the stages 
	while k < len(accelerometerData):

		# selct the accelerometer row at stage k
		row = selectRowByIndex(accelerometerData, k)

		# get the data
		accX = row.accelerometerX
		accY = row.accelerometerY
		accZ = row.accelerometerZ

		# local acceleration at time k 
		localAccQuat = [accX, accY, accZ, 0]

		# bring the acceleration into global frame of head
		globalAccQuat = accelerationToGlobalFrame(localAccQuat, orientation[k])

		# global acceleration vector
		globalAccVector = globalAccQuat[:3]

		# components of global acceleration vector
		[ax, ay, az] = globalAccVector

		# calculate the norm (overall rate of acceleration)
		norm = calculateNorm(ax, ay, az)

		# Axis of acceleration (i.e. normalise the acceleration)
		axis = normaliseVector(globalAccVector)

		# change in time
		changeInTime = timing[k] - timing[k-1]

		# Amount of acceleration during time delta t
		angle = norm * changeInTime
		
		# velocity change over time
		changeInVelocity = axisAngleToQuaternion(axis, angle)

		# estimated new velocity (as a quaternion)
		newVelocityQuat = quaternionProduct(changeInVelocity, velocityList[k-1])

		# add to list
		velocityList[k] = newVelocityQuat

		k+=1

	return velocityList

####################################################################
# v = rw --> w = v / r
def linearToAngularVelocity(linearVelocityList, radius):
	angularVelocityList = []
	for linearVelocity in linearVelocityList:
		angularVelocity = linearVelocity / radius
		angularVelocityList.append(angularVelocity)
	return angularVelocityList

####################################################################
# integrate angular velocity to get position 
def angularVelocityToOrientation(angularVelocityList):

	# estimated position list
	positionList = np.zeros((len(angularVelocityList), 4))

	# initial position is the identity quaternion 
	positionList[0] = [0,0,0,1]

	# get the time column 
	timing = getTimingInfo(data)

	k = 1

	# loop through all of the stages 
	while k < len(angularVelocityList):

		# get estimated angular velocity at stage k
		angularVelocityQuat = angularVelocityList[k]

		# new velocity as a vector (take first 3 values)
		angularVelocityVector = angularVelocityQuat[:3]

		# calculate the norm (overall rate of velocity)
		norm = calculateNorm(angularVelocityVector[0], angularVelocityVector[1], angularVelocityVector[2])

		# Axis of velocity (i.e. normalise the velocity)
		axis = normaliseVector(angularVelocityVector)

		# change in time
		changeInTime = timing[k] - timing[k-1]

		# Amount of velocity during time delta t
		angle = norm * changeInTime
			
		# position change over time
		changeInPosition = axisAngleToQuaternion(axis, angle)

		# estimated new position
		newPosition = quaternionProduct(changeInPosition, positionList[k-1])

		# add to list 
		positionList[k] = newPosition

		k+=1

	return positionList

####################################################################
# Positional tracking
# Distance from RCF to the base of the neck is 0.15m
# This is just using Equation 13 in LaValle's paper, following the kinematic constraint of
# Figure 10a
def estimatePositionMethod1(data, orientationList, length = 0.15):
	# estimate position from double integration of accelerometer
	firstPositionEstimate = accelerationToPosition(data)

	positionMap = []
	for orientation in orientationList:
		pos = headModelMapping(orientation, length)
		positionMap.append(pos)

	# add the two positions 
	position = [[sum(pair) for pair in zip(*pairs)] for pairs in zip(firstPositionEstimate, positionMap)]

	return position

####################################################################
# Length of torso = 0.5m
# Distance from RCF to the base of the neck is 0.15m
# This is using Equation 14 in LaValle's paper, following the kinematic constraint of
# Figure 10b
def estimatePositionMethod2(data, orientationList, length1 = 0.15, length2 = 0.4):

	# estimate position from double integration of accelerometer
	firstPositionEstimate = accelerationToPosition(data)

	# r from paper 
	r = []
	for orientation in orientationList:
		positionUpdate = headModelMapping(orientation, length1)
		r.append(positionUpdate)

	# get linear velocity via integration of linear acceleration
	linearVelocityList = accelerationToLinearVelocity(data, orientationList)

	# v = rw --> w = r / v
	angularVelocityList = linearToAngularVelocity(linearVelocityList, length2)

	# orientation list (i.e. q1 in the paper)
	orientationList2 = angularVelocityToOrientation(angularVelocityList)

	# r1 from paper 
	r1 = []
	for orientation in orientationList2:
		positionUpdate = headModelMapping(orientation, length2)
		r1.append(positionUpdate)
	
	# position map p = r1 + r
	# nested list comprehension element-wise additions
	position = [[sum(pair) for pair in zip(*pairs)] for pairs in zip(r, r1)]

	finalPosition = [[sum(pair) for pair in zip(*pairs)] for pairs in zip(r, r1)]

	return finalPosition

####################################################################
# update the vectors
def dataGen2(k, orientationList, positionList, lines):

	# get the orientation (as a quaternion) at stage k 
	orientation = orientationList[k]

	# get the position at at stage k 
	position = positionList[k]

	# Quaternion product of the unit vector representing an axis with the current orientation
	# This gives the DIRECTION of the axis at that timestep.
	# x,y,z components of the quaternion are a UNIT vector for the direction of the axis
	[x1, y1, z1, _] = quaternionVectorRotation([1,0,0,0], orientation)
	[x2, y2, z2, _] = quaternionVectorRotation([0,1,0,0], orientation)
	[x3, y3, z3, _] = quaternionVectorRotation([0,0,1,0], orientation)


	cartesianData = [[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]]

	# draw lines to point from origin
	# set_data sets X and Y positions 
	# set_3d_properties sets z-values
	# add the positions to the start and end points of the line
	for line, data in zip(lines, cartesianData):
		line.set_data([0+position[0], position[0]+data[0]], [0+position[1], position[1]+data[1]])
		line.set_3d_properties([0+position[2], position[2]+data[2]])

	return lines

####################################################################
# 3D plot for question 6
# plotting pose (orientation + position)
# this uses the function dataGen2 above to update the 3D plot
def create3DPosePlot(orientationData, positionData):
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.set_xlim3d(-1, 1)
	ax.set_ylim3d(-1, 1)
	ax.set_zlim3d(-1, 1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('3D Orientation Animation')

	# data = original orthogonal unit vector axis 
	data = [[[0,1], [0,0], [0,0]], [[0,0], [0,1], [0,0]], [[0,0], [0,0], [0,1]]]
	lines = [ax.plot(dat[0], dat[1], dat[2])[0] for dat in data]
	line_animation = animation.FuncAnimation(fig, dataGen2, fargs=(orientationData, positionData, lines), interval = 3.90625, blit = True)
	plt.show()

####################################################################
#### Running methods ####
####################################################################

firstEstimate = deadReckoningFilter(data)
#secondEstimate = tiltCorrection(firstEstimate, 0.04)
#thirdEstimate = yawCorrection(secondEstimate, 0.04)

# Plots for input dataset
#triple2DPlot1(data, unNormalisedAcceleromterData, unNormalisedMagnetometerData)

# 2D plots that show tri-axial Euler angles (degrees) for the 3 implemented methods
# (gyroscope integration, gyroscope + accelerometer, gyroscope + accelerometer + magnetometer)
#triple2DPlot2(firstEstimate, secondEstimate, thirdEstimate, True)


# 3D orientation plots
create3DOrientationPlot(firstEstimate, actualSpeed = True)
#create3DOrientationPlot(secondEstimat, actualSpeed = True)
#create3DOrientationPlot(thirdEstimate, actualSpeed = True)

sideBySide3D(firstEstimate, secondEstimate, thirdEstimate)

# Estimate position 
# Choose which method depending on kinematic constraints
#pos = estimatePositionMethod1(data, firstEstimate)
#pos = estimatePositionMethod2(data, firstEstimate)

# Q6 3D post plots (orientation + position)
#create3DPosePlot(firstEstimate, pos)

####################################################################

