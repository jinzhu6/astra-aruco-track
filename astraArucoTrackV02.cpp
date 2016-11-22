/// @file astraArucoTrackV02.cpp
///
/// @brief Tracks Aruco markers using the Astra S depth camera.
/// 
/// Markers are gouped and sorted by ID. Repeated IDs are averaged
/// by area weight. 3D joints are detected based on these marker
/// IDs. Maximum of 32 markers and maximum of 6 instances of 
/// each marker. Marker IDs must be 0-31.
///
/// Based on astraCalibratedDataV04.cpp. Writes first corner 
/// of marker 1 to file at 1 Hz. Press 'w' to write depth and 
/// color frames to image files as well as Stanford .ply file
/// for easy visualization. Also writes some tracking data to
/// log file.
///
/// TODO:
/// -fix limitation of 6
/// -data log write function with *fp as parameter
/// -joint map
///
/// Created 08 Oct 2016
/// Modified 11 Oct 2016
/// -sort by marker ID
/// -group repeated marker IDs
/// -marker area calc
/// -average repeated marker IDs
/// -3D joints lookup
///
/// @author Mustafa Ghazi
///

// for using printfs etc because MS has its own version
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#endif

#include <iterator> // because not included in Astra libraries by default
// Astra stuff
#include <Astra/Astra.h>
#include <AstraUL/AstraUL.h>
#include <cstdio>
#include <iostream>
// Open CV stuff
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// Aruco marker stuff
#include <opencv2/aruco.hpp>
using namespace std;
using namespace cv;

// frame width and height
#define DWIDTH 640
#define DHEIGHT 480
// depth range limits (mm) 500
#define MAXDEPTH 2000
#define MINDEPTH 500
// end program after this number of frames
#define NMAXFRAMES 18000
// for file names
#define METADATA "20161006_2350"
// no. of unique markers and max copies of each
#define NARUCOMARKERS 32
#define NARUCOCOPIES 6

// for frames of reference
enum FrameType {
	camBodyFrame,
	infantSupportFrame,
	infantBodyFrame
};

// for point in 3D space
struct Point3D {
	float x; ///< mm
	float y; ///< mm
	float z; ///< mm
};

// for inverse UV map
struct Map3D {
	int16_t u;			///< x-coordinate of color image (pixel)
	int16_t v;			///< y-coordinate of color image (pixel)
	int8_t isMapped;	///< 0 if not mapped, 1 if mapped
};

// for temporarily storing Aruco marker data
struct MarkerData {
	int16_t u;			///< x-coordinate of color image (pixel)
	int16_t v;			///< y-coordinate of color image (pixel)
	float area;			///< area of the quadrilateral (sq. pixel)	
};

// for joint locations
struct JointData {
	Point3D coordinates;///< 3D location in space
	FrameType frame;	///< frame of reference
	int8_t nSource;		///< no. of 2D markers used to detect
	int16_t u;			///< x-coordinate of color image (pixel)
	int16_t v;			///< y-coordinate of color image (pixel)
	float probability;	///< probability that this is a real joint	
};


Mat imgDepthMirroredCorrected = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // synthetic depth image, mirror corrected
Mat imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // image with color data
Mat imgColorMirrored = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // imgColor mirror corrected
Mat imgOverlay = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // color + depth overlay

Point3D DEPTHRAW[DWIDTH*DHEIGHT]; // xyz depth data
Map3D UVMAPINV[DWIDTH*DHEIGHT]; // map from color to depth pixels/data
char charCheckForKey = 0; // keyboard input
int DEPTHPT[2] = {0,0}; // read off depth from this point

/// @brief Return color intensity for depth value.
int lookupFunc (int depth);

/// @brief Update global flags and variables based on
/// global value that stores the last key stroke
void updateKeyCmds();

/// @brief Save color, depth, and overlay image view as a PNG 
/// image.
int saveImages();

/// @brief Save color, depth, and overlay image view as a 
/// Stanford .ply file.
int savePLYFile();

static void onMouse(int event, int x, int y, int f, void*);

/// @brief Generate an inverse UV map to lookup 3D coordinates 
/// given color pixels.
void generateInverseUVMap();

/// @brief Generate synthetic depth image based on 3D data
/// such that it is aligned with the color image.
void syntheticDepthImage (Mat dst);

/// @brief Parameter cx for a given depth.
float depthCorrectionLookup (float d);

/// @brief Generate synthetic blue, green, red values based on
/// depth value.
void fakeDepthColor(float depthVal, int8_t *blue, int8_t *green, int8_t *red);

/// @brief Get depth corresponding to a specific pixel.
float getDepthAt(int16_t x, int16_t y);

/// @brief Get 3D point corresponding to a specific pixel.
Point3D getPointAt(int16_t x, int16_t y);

/// @brief Sort detected Aruco markers by marker ID and compute 
/// some meta data.
void computeSortedMarkerMap(vector< vector<Point2f> > arucoCorners, vector< int > arucoIds,  int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES]);

/// @brief Compute the area of a quadrilateral given vector of 
/// 4 points.
float computeQuadrilateralArea(vector<Point2f> pts);

/// @brief Draw sorted markers on an image.
void drawSortedMarkerMap(int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES], Mat thisImage);

/// @brief Get average pixel locations in case multiple copies 
/// of a marker are detected.
void computePixelLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES], MarkerData AVG2DMARKERS[]);

/// @brief Draw average 2D markers on an image.
void drawPixelLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData AVG2DMARKERS[], Mat thisImage);

/// @brief Compute 3D locations of markers based on 2D pixel location.
void compute3DLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData AVG2DMARKERS[], JointData CURR3DMARKERS[]);

/// @brief Draw pixel location of 3D joints on an image.
void draw3DLocations(int SORTEDMARKERMAP[], int nMarkers, JointData CURR3DMARKERS[], Mat thisImage);

/// @brief Listens for and processes DepthPoint frames.
///
/// PointStream: World coordinates (XYZ) computed from the
/// depth data. The data array included in each PointFrame is
/// an array of astra:Vector3f elements to more easily access
/// the x, y and z values for each pixel.
///
class DepthFrameListener : public astra::FrameReadyListener
{
public:
	DepthFrameListener(int maxFramesToProcess) :
		m_maxFramesToProcess(maxFramesToProcess)
	{

	}

	bool is_finished()
	{
		return m_isFinished;
	}

private:
	/// @brief Do this when frame is available on stream reader
	///
	/// @param reader the stream reader
	/// @param frame the frame being read from the stream
	///
	/// @return Void
	///
	virtual void on_frame_ready(astra::StreamReader& reader,
		astra::Frame& frame) override
	{
		astra::PointFrame depthFrame = frame.get<astra::PointFrame>();

		if (depthFrame.is_valid())
		{
			processDepthFrame(depthFrame);
			++m_framesProcessed;
		}

		if (m_framesProcessed >= m_maxFramesToProcess)
		{
			m_isFinished = true;
		}
	}
	/// @brief Copy depth frame for further processing
	///
	/// @param depthFrame the frame to process
	///
	/// @return Void
	///
	void processDepthFrame(astra::PointFrame& depthFrame)
	{
		const astra::Vector3f* allFrameData = depthFrame.data();

		Vec3b intensity;
		int8_t blueVal, greenVal, redVal;
		int i;

		for(i=0;i<DWIDTH*DHEIGHT;i++) {

			DEPTHRAW[i].x = allFrameData[i].x;
			DEPTHRAW[i].y = allFrameData[i].y;
			DEPTHRAW[i].z = allFrameData[i].z;

		}
	}	
	bool m_isFinished{false};
	int m_framesProcessed{0};
	int m_maxFramesToProcess{0};
};


/// @brief Listens for and processes DepthPoint frames
///
/// Color Stream: RGB pixel data from the sensor. The data
/// array included in each ColorFrame contains values ranging
/// from 0-255 for each color component of each pixel.
///
class ColorFrameListener : public astra::FrameReadyListener
{
public:
	ColorFrameListener(int maxFramesToProcess) :
		m_maxFramesToProcess(maxFramesToProcess)
	{

	}

	bool is_finished()
	{
		return m_isFinished;
	}

private:
	/// @brief Do this when frame is available on stream reader
	///
	/// @param reader the stream reader
	/// @param frame the frame being read from the stream
	///
	/// @return Void
	///
	virtual void on_frame_ready(astra::StreamReader& reader,
		astra::Frame& frame) override
	{

		astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

		if (colorFrame.is_valid())
		{
			processColorFrame(colorFrame);
			++m_framesProcessed;
		}

		if (m_framesProcessed >= m_maxFramesToProcess)
		{
			m_isFinished = true;
		}
	}
	/// @brief Copy color frame for further processing
	///
	/// @param colorFrame the frame to process
	///
	/// @return Void
	///
	void processColorFrame(astra::ColorFrame& colorFrame)
	{
		const astra::RGBPixel* allFrameData = colorFrame.data();

		imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);
		Vec3b intensity;
		int i;

		for(i=0;i<DWIDTH*DHEIGHT;i++) {

			intensity.val[0] = allFrameData[i].b;	// B
			intensity.val[1] = allFrameData[i].g;	// G
			intensity.val[2] = allFrameData[i].r;	// R
			imgColor.at<Vec3b>(i/DWIDTH,i%DWIDTH) = intensity;
		}

	}

	bool m_isFinished{false};
	int m_framesProcessed{0};
	int m_maxFramesToProcess{0};
};


int main(int argc, char** arvg) {

	// setup Open CV 
	namedWindow("depth image", CV_WINDOW_AUTOSIZE);
	namedWindow("color image", CV_WINDOW_AUTOSIZE);
	setMouseCallback("depth image", onMouse, NULL);

	// setup Aruco marker dictionary to use
	int dictionaryId = aruco::DICT_4X4_100; // 4 x 4 squares, 100 qtty
	int markerId = 8; // 8 of the set of 0-199 (100 markers)
	int borderBits = 1; // default is 1, no. of squares at border
	int markerSize = 200; // size X size output image [pixels]
	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	// setup Aruco maker detection
	vector< int > markerIds; 
	vector< vector<Point2f> > markerCorners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create(); // default parameters

	// setup Orbbec Astra S camera
	int maxFramesToProcess = NMAXFRAMES;
	int iterationCounter = 0;
	double t = (double)getTickCount();
	astra::Astra::initialize();
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();
	reader.stream<astra::PointStream>().start(); // x,y,z
	reader.stream<astra::ColorStream>().start(); // R,G,B	
	DepthFrameListener listener(maxFramesToProcess);
	ColorFrameListener listener2(maxFramesToProcess);
	reader.addListener(listener);
	reader.addListener(listener2);

	// setup marker tracking and processing
	MarkerData SORTED2DMARKERS[NARUCOMARKERS][NARUCOCOPIES], AVG2DMARKERS[NARUCOMARKERS];
	JointData CURR3DMARKERS[NARUCOMARKERS];
	int SORTEDMARKERMAP[NARUCOMARKERS]; // memory map for sorted markers
	

	// setup datalogging
	char OUTPUT[75], METADATASTRING[25];
	sprintf(METADATASTRING, METADATA);
	sprintf(OUTPUT, "dataLog%s.txt",METADATASTRING);
	// file write stuff
	FILE * fpLog;
	fpLog = fopen (OUTPUT,"w");
	if(fpLog==NULL){
		printf("dataLog file write error!\n");
		return 1;
	}

	astra_temp_update(); // need to "pump" this

	while (!listener.is_finished() && !listener2.is_finished() && charCheckForKey != 27) {

		astra_temp_update(); // need to "pump" this

		flip(imgColor, imgColorMirrored, 1); // un-mirror color image
		generateInverseUVMap(); // inverse UV map for properly using depth data (required)
		syntheticDepthImage(imgDepthMirroredCorrected); // generate a synthetic depth image (for visualization)
		addWeighted(imgDepthMirroredCorrected, 0.5, imgColorMirrored, 0.5, 0.0, imgOverlay); // for visualization

		aruco::detectMarkers(imgColorMirrored, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);
		aruco::drawDetectedMarkers(imgColorMirrored, markerCorners, markerIds); 
		aruco::drawDetectedMarkers(imgOverlay, markerCorners, markerIds); 

		
		// process marker data
		computeSortedMarkerMap(markerCorners, markerIds, SORTEDMARKERMAP, NARUCOMARKERS, SORTED2DMARKERS);
		//drawSortedMarkerMap(SORTEDMARKERMAP, NARUCOMARKERS, SORTED2DMARKERS, imgColorMirrored);
		computePixelLocations(SORTEDMARKERMAP, NARUCOMARKERS, SORTED2DMARKERS, AVG2DMARKERS);
		//drawPixelLocations(SORTEDMARKERMAP, NARUCOMARKERS, AVG2DMARKERS, imgColorMirrored);
		compute3DLocations(SORTEDMARKERMAP,NARUCOMARKERS,AVG2DMARKERS,CURR3DMARKERS);
		draw3DLocations(SORTEDMARKERMAP, NARUCOMARKERS, CURR3DMARKERS, imgColorMirrored);

		char TEXT[32]; // for displaying marker range
		sprintf(TEXT,"%dmm",(int)getDepthAt(DEPTHPT[0],DEPTHPT[1]));
		putText(imgColorMirrored, TEXT, Point(25,25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,192,55), 2, 8, false );

		char TEXT2[56]; // for displaying xyz
		Point3D currData = getPointAt(DEPTHPT[0],DEPTHPT[1]);
		sprintf(TEXT2,"%d,%d,%d",(int)currData.x,(int)currData.y,(int)currData.z);
		putText(imgColorMirrored, TEXT2, Point(25,55), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,192,55), 2, 8, false );

		imshow("depth image", imgDepthMirroredCorrected);
		imshow("color image", imgColorMirrored);

		iterationCounter++;

		// print frame rate, depth info every 30 frames
		if(iterationCounter%5 == 0) {

			t = ((double)getTickCount() - t)/getTickFrequency();
			int i;
			
			if((markerIds.size()==2)&&(CURR3DMARKERS[0].nSource == 1)&&(CURR3DMARKERS[1].nSource == 1)) {
				float e1 = abs(CURR3DMARKERS[0].coordinates.x) + abs(CURR3DMARKERS[0].coordinates.y) + abs(CURR3DMARKERS[0].coordinates.z); // error metric to check if point is (0,0,0)
				float e2 = abs(CURR3DMARKERS[1].coordinates.x) + abs(CURR3DMARKERS[1].coordinates.y) + abs(CURR3DMARKERS[1].coordinates.z);
				if((e1>3.0) && (e2>3.0)) {
					printf("0,%.2f,%.2f,%.2f,",CURR3DMARKERS[0].coordinates.x,CURR3DMARKERS[0].coordinates.y,CURR3DMARKERS[0].coordinates.z);
					printf("1,%.2f,%.2f,%.2f\n",CURR3DMARKERS[1].coordinates.x,CURR3DMARKERS[1].coordinates.y,CURR3DMARKERS[1].coordinates.z);
					float distance = (CURR3DMARKERS[0].coordinates.x - CURR3DMARKERS[1].coordinates.x)*(CURR3DMARKERS[0].coordinates.x - CURR3DMARKERS[1].coordinates.x);
					distance = distance + (CURR3DMARKERS[0].coordinates.y - CURR3DMARKERS[1].coordinates.y)*(CURR3DMARKERS[0].coordinates.y - CURR3DMARKERS[1].coordinates.y);
					distance = distance + (CURR3DMARKERS[0].coordinates.z - CURR3DMARKERS[1].coordinates.z)*(CURR3DMARKERS[0].coordinates.z - CURR3DMARKERS[1].coordinates.z);
					distance = sqrt(distance);
					//printf("dist = %.2f\n",distance);
				}
				
			}
			/*
			if(markerIds.size()>0) {
				for(i=0;i<markerIds.size();i++) {
					if(markerIds[i] == 1) {
						fprintf(fpLog,"%d,%.2f,%.2f\n",markerIds[i], markerCorners[i][0].x, markerCorners[i][0].y);

					}

				}
			}
			printf("t= %f, (%d,%d)\n",t/30.0, DEPTHPT[0], DEPTHPT[1]);
			*/
			t = (double)getTickCount();
		}

		charCheckForKey = waitKey(30);
		updateKeyCmds();
	}

	reader.removeListener(listener);
	reader.removeListener(listener2);
	astra::Astra::terminate();
	fclose(fpLog);
	printf("Saved data log file successfully!\n");

	return 0;
}


/// @brief Return color intensity for depth value.
///
/// Intensity is generated using a triangular function with
/// peak at (255,255) and end points (0,0) and (510,0). This
/// is useful in generating an R/G/B color value for a number
/// between 0 and 510, which peaks at 255. With 3 color
/// channels, it can be used so that different colors fade
/// in and out for different depth values.
///
/// Returns 0 outside the 0-510 range. So the provided depth
/// value must always be scaled/shifted to fit this range.
///
/// @param depth the depth value for which to generate color
///
/// @return a color value from 0-255
///
int lookupFunc (int depth) {

	if(depth < 0) {
		return  0;
	} else if( (depth >=0) && (depth <=255)) {
		return depth;
	} else if( (depth > 255) && (depth <= 510) ) {
		return (510 - depth);
	} else {
		return 0;
	}

}


/// @brief Update global flags and variables based on
/// global value that stores the last key stroke
///
/// @return Void
///
void updateKeyCmds() {

	if(charCheckForKey == 'w') { 
		saveImages();
		savePLYFile();
	}
}


/// @brief Save color, depth, and overlay image view as a PNG 
/// image.
///
/// @return 0 if read successfully, 1 if failed
///
int saveImages() {

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	char OUTPUT1[75], METADATASTRING1[25], OUTPUT2[75], METADATASTRING2[25], OUTPUT3[75], METADATASTRING3[25];
	sprintf(METADATASTRING1, METADATA);
	sprintf(OUTPUT1, "orbbecColor%s.png",METADATASTRING1);
	sprintf(METADATASTRING2, METADATA);
	sprintf(OUTPUT2, "orbbecDepth%s.png",METADATASTRING2);
	sprintf(METADATASTRING3, METADATA);
	sprintf(OUTPUT3, "orbbecFused%s.png",METADATASTRING3);
	try {
		imwrite(OUTPUT1, imgColorMirrored, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}

	try {
		imwrite(OUTPUT2, imgDepthMirroredCorrected, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}

	try {
		imwrite(OUTPUT3, imgOverlay, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}
	fprintf(stdout, "Saved PNG file for color image window.\n");
}


/// @brief Save color, depth, and overlay image view as a 
/// Stanford .ply file.
///
/// @return 0 if read successfully, 1 if failed
///
int savePLYFile() {

	printf("Attempting to write .ply file...\n");
	int i=0, iDepth=0, counter=0;
	Vec3b intensity;

	char OUTPUT[75], METADATASTRING[25];
	sprintf(METADATASTRING, METADATA);
	sprintf(OUTPUT, "orbbecPoints%s.ply",METADATASTRING);

	// file write stuff
	FILE * fp;
	fp = fopen (OUTPUT,"w");
	if(fp==NULL){
		printf(".ply file write error!\n");
		return 1;
	}

	// count elements
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			counter++;
		}
	}

	// header info
	fprintf(fp,"ply\nformat ascii 1.0\ncomment author: Mustafa Ghazi, Intelligent Robotics Lab\ncomment object: Orbbec Astra S data\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",counter);

	// cycle through color pixels
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			iDepth = DWIDTH*UVMAPINV[i].v + UVMAPINV[i].u;
			intensity = imgColorMirrored.at<Vec3b>(i/DWIDTH,i%DWIDTH); // y,x
			fprintf(fp,"%.2f %.2f %.2f %d %d %d\n",DEPTHRAW[iDepth].x,DEPTHRAW[iDepth].y,DEPTHRAW[iDepth].z,intensity.val[2],intensity.val[1],intensity.val[0]);
		}
	}

	fclose(fp);
	printf("Saved .ply file successfully!\n");

	return 0;
}


static void onMouse(int event, int x, int y, int f, void*) {

	if (event == CV_EVENT_MOUSEMOVE) {
		DEPTHPT[0] = x; 
		DEPTHPT[1] = y;
	}
}


/// @brief Generate an inverse UV map to lookup 3D coordinates 
/// given color pixels.
///
/// Every i-th elemnt in the UV map array corresponds to the 
/// i-th element (i%WIDTH,i/WIDTH) in the color image. The 
/// (u,v) values represent the corresponding depth image pixel
/// (i=u+v*WIDTH). The variable isMapped indicates whether
/// this array element is mapped or not.
///
/// An element is mapped only if its z-depth is between MINDEPTH
/// and MAXDEPTH. 
///
/// @return Void
///
void generateInverseUVMap() {

	int i, xColor, yColor, xDepth, yDepth;
	int16_t currDepth;
	float kx = 0.887, cx = 15.33, ky = 0.8837, cy = 18; // default depth correction parameters
														// clear inverse UV map
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		UVMAPINV[i].isMapped = 0; // set to unmapped
	}
	// go through the depth image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		currDepth = (int16_t)DEPTHRAW[i].z;
		if(currDepth>=MINDEPTH && currDepth<=MAXDEPTH) {
			xDepth = i%DWIDTH;
			yDepth = i/DWIDTH;
			xColor = (float)(DWIDTH-xDepth)*kx + depthCorrectionLookup(currDepth); // map the mirror corrected x-coord
			yColor = (float)(yDepth)*ky + cy;
			UVMAPINV[yColor*DWIDTH+xColor].isMapped = 1; // in range, set to mapped
			UVMAPINV[yColor*DWIDTH+xColor].u = xDepth;
			UVMAPINV[yColor*DWIDTH+xColor].v = yDepth;

		}
	}
}


/// @brief Generate synthetic depth image based on 3D data
/// such that it is aligned with the color image.
///
/// Image is generated based on inverse UV Map. Colors are
/// limited to the predefined min/max depth range.
///
/// @param dst generated image
///
/// @return Void
///
void syntheticDepthImage(Mat dst) {

	int i, iDepth;
	int8_t blue, green, red;
	float thisDepth;
	Vec3b intensity;
	dst = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);	

	// go through the indices corresponding to the color image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			iDepth = DWIDTH*UVMAPINV[i].v + UVMAPINV[i].u;
			fakeDepthColor(DEPTHRAW[iDepth].z,&blue,&green,&red);
			intensity.val[0] = blue;
			intensity.val[1] = green;
			intensity.val[2] = red;

			dst.at<Vec3b>(i/DWIDTH,i%DWIDTH) = intensity;
		}
	}
}


/// @brief Parameter cx for a given depth.
///
/// This is for the alignment of depth image pixels to color
/// image pixels for this model:
/// x_adjusted = kx*x + cx
/// y_adjusted = ky*y + cy
///
/// As of 02 Oct 2016 it is valid for ~550 to ~1724 mm.
///
/// @param depth (mm)
///
/// @return cx (pixel)
///
float depthCorrectionLookup (float depth) {
	float cx;
	int d = (int)depth;
	if(d>=550 && d<=780) {
		cx = 0.043478*depth - 11.913;
	} else if(d>780 && d<=1020) {
		cx = 0.016667*depth + 8.9997;
	} else if(d>1020 && d<=1724) {
		cx = 0.0042614*depth + 21.653;
	} else if(d>1724) {
		cx = 29; // out of bounds of model
	} else if(d<550) {
		cx = 12; // out of bounds of model
	}

	return cx;
}


/// @brief Generate synthetic blue, green, red values based on
/// depth value.
///
/// @param depthVal depth value [mm]
/// @param *blue blue channel value in range 0-255
/// @param *green green channel value in range 0-255
/// @param *red red channel value in range 0-255
///
/// @return Void
///
void fakeDepthColor(float depthVal, int8_t *blue, int8_t *green, int8_t *red) {

	*blue = (int8_t)lookupFunc((int16_t)depthVal/3-600);	// B
	*green = (int8_t)lookupFunc((int16_t)depthVal/3-300);	// G
	*red = (int8_t)lookupFunc((int16_t)depthVal/3);			// R

}


/// @brief Get depth corresponding to a specific pixel.
///
/// x and y must lie within frame width and frame height.
///
/// Coordinate systems:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
/// -Uncorrected depth camera image frame, origin at top left
/// corner, +u right, +v down
/// -camera body coordinate frame, origin at ?? +z out the 
/// image plane, +x left +y up
///
/// @param x x-coordinate, mirror corrected color camera image
/// frame [pixels]
/// @param y y-coordinate, mirror corrected color camera image
/// frame [pixels]
///
/// @return z coordinate, camera body frame
///
float getDepthAt(int16_t x, int16_t y) {

	// check if within range	
	if((x<0) || (x>DWIDTH) || (y<0) || (y>DHEIGHT)) {
		return 0.0;
	}

	int clrIdx = DWIDTH*y + x;
	int depthIdx = DWIDTH*UVMAPINV[clrIdx].v  + UVMAPINV[clrIdx].u;	
	if((depthIdx >=0) && (depthIdx<=DWIDTH*DHEIGHT)){
		return DEPTHRAW[depthIdx].z;
	} else { 
		return 0.0; 
	}
}


/// @brief Get 3D point corresponding to a specific pixel.
///
/// x and y must lie within frame width and frame height.
///
/// Coordinate systems:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
/// -Uncorrected depth camera image frame, origin at top left
/// corner, +u right, +v down
/// -camera body coordinate frame, origin at ?? +z out the 
/// image plane along LOS, +x left +y up
///
/// @param x x-coordinate, mirror corrected color camera image
/// frame [pixels]
/// @param y y-coordinate, mirror corrected color camera image
/// frame [pixels]
///
/// @return Point3D, camera body frame
///
Point3D getPointAt(int16_t x, int16_t y) {

	Point3D thisPoint;
	thisPoint.x = 0;
	thisPoint.y = 0;
	thisPoint.z = 0;
	// check if within range	
	if((x<0) || (x>DWIDTH) || (y<0) || (y>DHEIGHT)) {
		return thisPoint;
	}

	int clrIdx = DWIDTH*y + x;
	int depthIdx = DWIDTH*UVMAPINV[clrIdx].v  + UVMAPINV[clrIdx].u;	
	if((depthIdx >=0) && (depthIdx<=DWIDTH*DHEIGHT)){
		thisPoint.x = DEPTHRAW[depthIdx].x;
		thisPoint.y = DEPTHRAW[depthIdx].y;
		thisPoint.z = DEPTHRAW[depthIdx].z;
		return thisPoint;
	} else { 
		return thisPoint; 
	}
}


/// @brief Sort detected Aruco markers by marker ID and compute 
/// some meta data.
///
/// A sorted marker map is a 1D array indicating how many 
/// Aruco markers have been detected for each Aruco dictionary
/// ID. The index of each array element is the dictionary ID,
/// e.g. index 0 corresponds to ID 0.
///
/// As each detected marker is processed and stored into the 
/// marker data struct, the count in the marker map is updated for
/// the corresponding dictionary ID.
///
/// Meta data include the center point and area of each marker.
/// These data are saved for each detected copy of the marker.
///
/// Parameter validation includes a check for marker ID being 
/// in range and number of marker corners being exactly 4.
///
/// Coordinate frame:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
///
/// @param arucoCorners marker corners as detected by Aruco library
/// @param arucoIds corresponding marker IDs 
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param SORTED2DMARKERS marker data sorted by marker ID
///
/// @return Void
///
void computeSortedMarkerMap(vector< vector<Point2f> > arucoCorners, vector< int > arucoIds,  int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES]) {

	int i, area, idx;
	float cx, cy;
	
	// clear the marker map
	for(i=0;i<nMarkers;i++) {
		SORTEDMARKERMAP[i] = 0; // n markers selected = 0
	}

	for(i=0;i<arucoIds.size();i++) {
		// make sure there are 4 corners and ID is within range
		if( (arucoCorners[i].size() == 4)  && (arucoIds[i] < nMarkers) ){
			cx = (float)(arucoCorners[i][0].x + arucoCorners[i][1].x + arucoCorners[i][2].x + arucoCorners[i][3].x)/4.0;
			cy = (float)(arucoCorners[i][0].y + arucoCorners[i][1].y + arucoCorners[i][2].y + arucoCorners[i][3].y)/4.0;
			area = computeQuadrilateralArea(arucoCorners[i]);
			idx = SORTEDMARKERMAP[arucoIds[i]]; // new idx = last count
			SORTED2DMARKERS[arucoIds[i]][idx].u = (int)cx;
			SORTED2DMARKERS[arucoIds[i]][idx].v = (int)cy;
			SORTED2DMARKERS[arucoIds[i]][idx].area = area;
			idx++; // update the count
			
			SORTEDMARKERMAP[arucoIds[i]] = idx; 
		}
	}
}


/// @brief Compute the area of a quadrilateral given vector of 
/// 4 points.
///
/// Aruco marker points are provided clockwise. Uses shoelace
/// formula, also known as Gauss's area formula, or surveyor's 
/// formula.
///
/// @param pts (pixels)
///
/// @return area of quadrilateral (sq. pixels)
///
float computeQuadrilateralArea(vector<Point2f> pts) {

	// to fix
	float sum;
	if(pts.size()==4) {
		sum = std::abs(pts[0].x*pts[1].y + pts[1].x*pts[2].y + pts[2].x*pts[3].y + pts[3].x*pts[0].y - pts[1].x*pts[0].y - pts[2].x*pts[1].y - pts[3].x*pts[2].y - pts[0].x*pts[3].y);
		return (int)(0.5*sum);
	} else {
		return 1;
	}
}


/// @brief Draw sorted markers on an image.
///
/// For each marker, draws a circle proportional to its area.
///
/// Coordinate frame:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
///
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param SORTED2DMARKERS marker data sorted by marker ID
/// @param thisImage image on which the markers are to be drawn
///
/// @return Void
///
void drawSortedMarkerMap(int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES], Mat thisImage) {

	int i, j;

	for(i=0;i<nMarkers;i++) {
		if(SORTEDMARKERMAP[i] > 0) {
			for(j=0;j<SORTEDMARKERMAP[i];j++) {
				circle(thisImage, Point(SORTED2DMARKERS[i][j].u,SORTED2DMARKERS[i][j].v), (int)(0.5*sqrt(SORTED2DMARKERS[i][j].area)), Scalar(210, 71, 30), 2, LINE_8, 0);
			}
		}
	}

}


/// @brief Get average pixel locations in case multiple copies 
/// of a marker are detected.
///
/// Location for a marker ID is a weighted average of the 
/// locations of all the detected copies of that marker ID. The
/// average is weighted by individual marker areas.
///
/// No further sorting is performed. 
///
/// Coordinate frame:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
///
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param SORTED2DMARKERS marker data sorted by marker ID
/// @param AVG2DMARKERS average of the sorted marker data
///
/// @return Void
///
void computePixelLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData SORTED2DMARKERS[][NARUCOCOPIES], MarkerData AVG2DMARKERS[]) {

	int i, j, count;
	float cx, cy, divide;
	for(i=0;i<nMarkers;i++) {
		if(SORTEDMARKERMAP[i] > 0) {
			cx = 0; 
			cy = 0;
			divide = 0;
			count = SORTEDMARKERMAP[i];
			for(j=0;j<count;j++) {
				divide = divide + SORTED2DMARKERS[i][j].area; // total weight
				cx = cx + SORTED2DMARKERS[i][j].area*SORTED2DMARKERS[i][j].u; // weighted sum
				cy = cy + SORTED2DMARKERS[i][j].area*SORTED2DMARKERS[i][j].v; // weighted sum
			}
			AVG2DMARKERS[i].u = (int16_t)(cx/divide); // weighted average
			AVG2DMARKERS[i].v = (int16_t)(cy/divide); // weighted average
		} else {
			AVG2DMARKERS[i].u = 0; // clear the data
			AVG2DMARKERS[i].v = 0;
		}
	}

}


/// @brief Draw average 2D markers on an image.
///
/// Draws circles of fixed radius.
///
/// Coordinate frame:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
///
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param AVG2DMARKERS average of the sorted marker data
/// @param thisImage image on which the markers are to be drawn
///
/// @return Void
///
void drawPixelLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData AVG2DMARKERS[], Mat thisImage){

	int i;

	for(i=0;i<nMarkers;i++) {
		if(SORTEDMARKERMAP[i] > 0) {
			circle(thisImage, Point(AVG2DMARKERS[i].u,AVG2DMARKERS[i].v), 18, Scalar(32, 155, 238), 2, LINE_8, 0);

		}
	}

}


/// @brief Compute 3D locations of markers based on 2D pixel location.
///
/// Other meta data is also computed.
///
/// Coordinate frames:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
/// -camera body coordinate frame, origin at ?? +z out the 
/// image plane along LOS, +x left +y up
///
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param AVG2DMARKERS average of the sorted marker data
/// @param CURR3DMARKERS 3D locations of detected markers
///
/// @return Void
///
void compute3DLocations(int SORTEDMARKERMAP[], int nMarkers, MarkerData AVG2DMARKERS[], JointData CURR3DMARKERS[]) {

	int i;

	for(i=0;i<nMarkers;i++) {
		if(SORTEDMARKERMAP[i] > 0) {

			CURR3DMARKERS[i].nSource = SORTEDMARKERMAP[i];
			CURR3DMARKERS[i].u = AVG2DMARKERS[i].u;
			CURR3DMARKERS[i].v = AVG2DMARKERS[i].v;
			CURR3DMARKERS[i].probability = 1.0;
			CURR3DMARKERS[i].frame = camBodyFrame;
			CURR3DMARKERS[i].coordinates = getPointAt(CURR3DMARKERS[i].u, CURR3DMARKERS[i].u);

		} else {

			CURR3DMARKERS[i].nSource = 0;
			CURR3DMARKERS[i].u = 0;
			CURR3DMARKERS[i].v = 0;
			CURR3DMARKERS[i].probability = 0.0;
			CURR3DMARKERS[i].frame = camBodyFrame;
			CURR3DMARKERS[i].coordinates.x = 0;
			CURR3DMARKERS[i].coordinates.y = 0;
			CURR3DMARKERS[i].coordinates.z = 0;
		}
	}

}


/// @brief Draw pixel location of 3D joints on an image.
///
/// Draws circles of fixed radius.
///
/// Coordinate frames:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
///
/// @param SORTEDMARKERMAP marker map
/// @param nMarkers no. of unique Aruco markers
/// @param CURR3DMARKERS 3D locations of detected markers
/// @param thisImage image on which the markers are to be drawn
///
/// @return Void
///
void draw3DLocations(int SORTEDMARKERMAP[], int nMarkers, JointData CURR3DMARKERS[], Mat thisImage){

	int i;

	for(i=0;i<nMarkers;i++) {
		if(SORTEDMARKERMAP[i] > 0) {
			circle(thisImage, Point(CURR3DMARKERS[i].u,CURR3DMARKERS[i].v), 22, Scalar(82,220,40), 2, LINE_8, 0);

		}
	}

}