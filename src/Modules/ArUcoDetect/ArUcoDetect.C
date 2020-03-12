// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevoisbase/Components/ArUco/ArUco.H>
#include <opencv2/core/core.hpp>

#define ARUCO_MSG_ID 0x41
#define ARUCO_SYNC1 0x31
#define ARUCO_SYNC2 0x32
#define ARUCO_SYNC3 0x33

struct msg_header{
    unsigned char sync1; /*  */
    unsigned char sync2; /*  */
    unsigned char sync3; /*  */
    unsigned char spare; /*  */
    int messageID; /* id # */
    int messageSize; /* including header */
    unsigned int hcsum; /*  */
    unsigned int csum; /*  */
}__attribute__((packed));

struct ArUco_msg {
  ArUco_msg() {ArUco_header.sync1    = ARUCO_SYNC1;
              ArUco_header.sync2     = ARUCO_SYNC2;
              ArUco_header.sync3     = ARUCO_SYNC3;
              ArUco_header.messageID = ARUCO_MSG_ID;}
  msg_header ArUco_header;
  int count; /*  */
  float time; /* time (sec) */
  float py; /* pixel y location (+ right) */
  float pz; /* pixel z location (+ down) */
  float psqrtA; /* pixel square root of Area */
  float confidence; /* confidence in solution */
  float fps; /* frames processed per second */
}__attribute__((packed));


/* the checksum is just for the message body, skipping the header (according to datalink.cpp in GUST) */
unsigned int calculateCheckSum(unsigned char *buf, int byteCount) {

	unsigned int sum1 = 0xffff;
	unsigned int sum2 = 0xffff;
	unsigned int tlen = 0;
	unsigned int shortCount = byteCount / sizeof(short);
	unsigned int oddLength = byteCount % 2;

	/* this is Fletcher32 checksum modified to handle buffers with an odd number of bytes */

	while (shortCount) {
		/* 360 is the largest number of sums that can be performed without overflow */
		tlen = shortCount > 360 ? 360 : shortCount;
		shortCount -= tlen;
		do {
			sum1 += *buf++;
			sum1 += (*buf++ << 8);
			sum2 += sum1;
		} while (--tlen);

		/* add last byte if there's an odd number of bytes (equivalent to appending a zero-byte) */
		if ((oddLength == 1) && (shortCount < 1)) {
			sum1 += *buf++;
			sum2 += sum1;
		}

		sum1 = (sum1 & 0xffff) + (sum1 >> 16);
		sum2 = (sum2 & 0xffff) + (sum2 >> 16);
	}

	/* Second reduction step to reduce sums to 16 bits */
	sum1 = (sum1 & 0xffff) + (sum1 >> 16);
	sum2 = (sum2 & 0xffff) + (sum2 >> 16);

	return(sum2 << 16 | sum1);
}

std::string encodeSerialMsg(char *buf, int byteCount){
    std::string msg;
    for (int i = 0; i < byteCount; i++){
        msg += buf[i];
    }
    return msg;
}


#define IMGHHEIGHT 1024
#define IMGWIDTH 1280

// icon by Catalin Fertu in cinema at flaticon


//! Simple demo of ArUco augmented reality markers detection and decoding
/*! Detect and decode patterns known as ArUco markers, which are small 2D barcodes often used in augmented
    reality and robotics.

    ArUco markers are small 2D barcodes. Each ArUco marker corresponds to a number, encoded into a small grid of black
    and white pixels. The ArUco decoding algorithm is capable of locating, decoding, and of estimating the pose
    (location and orientation in space) of any ArUco markers in the camera's field of view.

    ArUco markers are very useful as tags for many robotics and augmented reality applications. For example, one may
    place an ArUco next to a robot's charging station, an elevator button, or an object that a robot should manipulate.

    For more information about ArUco, see https://www.uco.es/investiga/grupos/ava/node/26

    The implementation of ArUco used by JeVois is the one of OpenCV-Contrib, documented here:
    http://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html

    ArUco markers can be created with several standard dictionaries. Different dictionaries give rise to different
    numbers of pixels in the markers, and to different numbers of possible symbols that can be created using the
    dictionary. The default dictionary used by JeVois is 4x4 with 50 symbols. Other dictionaries are also supported by
    setting the parameter \p dictionary over serial port or in a config file, up to 7x7 with 1000 symbols.

    Creating and printing markers
    -----------------------------

    We have created the 50 markers available in the default dictionary (4x4_50) as PNG images that you can download and
    print, at http://jevois.org/data/ArUco.zip

    To make your own, for example, using another dictionary, see the documentation of the \ref ArUco component of
    JeVoisBase. Some utilities are provided with the component.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle.

    When \p dopose is turned on, 3D messages will be sent, otherwise 2D messages.

    One message is issued for every detected ArUco, on every video frame.

    2D messages when \p dopose is off:

    - Serial message type: \b 2D
    - `id`: decoded ArUco marker ID, with a prefix 'U'
    - `x`, `y`, or vertices: standardized 2D coordinates of marker center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string)

    3D messages when \p dopose is on:

    - Serial message type: \b 3D
    - `id`: decoded ArUco marker ID, with a prefix 'U'
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of marker center or corners
    - `w`, `h`, `d`: marker size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string)

    If you will use the quaternion data (Detail message style; see \ref UserSerialStyle), you should probably set the \p
    serprec parameter to something non-zero to get enough accuracy in the quaternion values.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    Things to try
    -------------

    - First, use a video viewer software on a host computer and select one of the video modes with video output over
      USB. Point your JeVois camera towards one of the screenshots provided with this module, or towards some ArUco
      markers that you find on the web or that you have printed from the collection above (note: the default dictionary
      is 4x4_50, see parameter \p dictionary).

    - Then try it with no video output, as it would be used by a robot. Connect to the command-line interface of your
      JeVois camera through the serial-over-USB connection (see \ref UserCli; on Linux, you would use <b>sudo screen
      /dev/ttyACM0 115200</b>) and try:
      \verbatim
      setpar serout USB
      setmapping2 YUYV 320 240 30.0 JeVois DemoArUco
      streamon
      \endverbatim
      and point the camera to some markers; the camera should issue messages about all the markers it identifies.

    Computing and showing 3D pose
    -----------------------------

    The OpenCV ArUco module can also compute the 3D location and orientation of each marker in the world when \p dopose
    is true. The requires that the camera be calibrated, see the documentation of the \ref ArUco component in
    JeVoisBase. A generic calibration that is for a JeVois camera with standard lens is included in files \b
    calibration640x480.yaml, \b calibration352x288.yaml, etc in the jevoisbase share directory (on the MicroSD, this is
    in <b>JEVOIS:/share/camera/</b>).

    When doing pose estimation, you should set the \p markerlen parameter to the size (width) in millimeters of your
    actual physical markers. Knowing that size will allow the pose estimation algorithm to know where in the world your
    detected markers are.

    For more about camera calibration, see [this tutorial](http://jevois.org/tutorials/UserArUcoCalib.html) and
    http://jevois.org/basedoc/ArUco_8H_source.html

    Tutorial and video
    ------------------

    Check out this tutorial on how to [build a simple visually-guided toy robot car for under $100 with
    JeVois](http://jevois.org/tutorials/UserRobotCar.html), which uses ArUco at its core. A demo video is here:

    \youtube{7cMtD-ef83E}


    @author Laurent Itti
    Edited by Oliver Dunbabin

    @displayname Demo ArUco
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DemoArUco
    @videomapping YUYV 320 260 30.0 YUYV 320 240 30.0 JeVois DemoArUco
    @videomapping YUYV 640 500 20.0 YUYV 640 480 20.0 JeVois DemoArUco
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2016 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class ArUcoDetect : public jevois::StdModule
{
  public:
    //! Constructor
    ArUcoDetect(std::string const & instance): jevois::StdModule(instance)
    {
        itsArUco = addSubComponent<ArUco>("aruco");
    }
    
    //! Virtual destructor for safe inheritance
    virtual ~ArUcoDetect() { }

    //! Processing function, no video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      static jevois::Timer timer("processing", 199, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);

      timer.start();

      unsigned int const w = inimg.width, h = inimg.height;

      // Convert the image to grayscale
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f> > corners;
      itsArUco->detectMarkers(cvimg, ids, corners);
      
      // Do pose computations
      std::vector<cv::Vec3d> rvecs, tvecs;
      if (ids.empty() == false)
          itsArUco->estimatePoseSingleMarkers(corners, rvecs, tvecs);

      // Let camera know we are done processing the input image:
      inframe.done(); // NOTE: optional here, inframe destructor would call it anyway

      std::string const &fpscpu = timer.stop();

      for (size_t i = 0; i < ids.size(); i++){
          std::vector<cv::Point2f> tag = corners[i];
          // Compute C.G. of tag
          float cy = 0.0, cz = 0.0;
          std::vector<cv::Point2f> vertices;
          for (cv::Point2f const & p : tag)
          {
              cy += p.x; cz += p.y;
              cv::Point2f pix;
              pix.x = (p.x - 0.5*(float)w)/(0.5*(float)w)*(h/(double)w);
              pix.y = (p.y - 0.5*(float)h)/(0.5*(float)w)*(h/(double)w);
              vertices.push_back(pix);
          }
          // Calculate area of ArUco code
          size_t n = vertices.size();
          int j = n - 1;
          float area = 0;
          for (size_t i = 0; i < n; i++)
          {
              area += (vertices[j].x + vertices[i].x)*(vertices[j].y - vertices[i].y);
              j = i;
          }
          if (n) { cy /= n; cz /= n; area = abs(area/2.0); }

          // Pack message
          ArUco_msg msg;
          msg.py        = (cy - 0.5*(float)w)/(0.5*(float)w)*(h/((double)w));  // or *(IMGHEIGHT/IMGWIDTH)
          msg.pz        = (cz - 0.5*(float)h)/(0.5*(float)w)*(h/((double)w));  // or *(IMGHEIGHT/IMGWIDTH)
          msg.time      = inimg.time_stmp.tv_sec + inimg.time_stmp.tv_usec/1000000.;
          msg.psqrtA    = sqrt(area);

          // Encode message
          msg.ArUco_header.messageSize = sizeof(struct ArUco_msg);
          int byteCount         = msg.ArUco_header.messageSize;
          int headerSize        = sizeof(struct msg_header);
          int index             = headerSize;

		  /* csum is the checksum of just the message content (ignoring the header) */
          unsigned int csum     = calculateCheckSum((unsigned char *)&msg[sizeof(struct msg_header)], (int)(sizeof(struct ArUco_msg) - sizeof(struct msg_header)));
          
		  /* hcsum is the checksum of just the header (ignoring the content) -- I don't think GUST checks this... */
		  unsigned int hcsum    = calculateCheckSum((unsigned char *)&msg, sizeof(struct msg_header));
          
		  msg.ArUco_header.csum = csum;
          msg.ArUco_header.hcsum= hcsum;
          std::string ArUco_string = encodeSerialMsg((char *)&msg, byteCount);
          jevois::Module::sendSerial(ArUco_string);
      }
    }


    //! Processing function with output video to USB
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {

      static jevois::Timer timer("processing", 199, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);

      timer.start();

      unsigned int const w = inimg.width, h = inimg.height;

      // We only support YUYV pixels in this example, any resolution:
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&](){
            outimg = outframe.get();
            outimg.require("output", w, h+20, inimg.fmt);
            jevois::rawimage::paste(inimg, outimg, 0, 0);
            jevois::rawimage::writeText(outimg, "Detecting ArUco", 3, 3, jevois::yuyv::White);
            jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
        });

      // Convert the image to grayscale and process
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      std::vector<cv::Vec3d> rvecs, tvecs;
      itsArUco->detectMarkers(cvimg, ids, corners);

      if (ids.empty() == false)
          itsArUco->estimatePoseSingleMarkers(corners, rvecs, tvecs);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();
      
      //Show results
      itsArUco->drawDetections(outimg, 3, h+5, ids, corners, rvecs, tvecs);

      std::string const &fpscpu = timer.stop();

      for (size_t i = 0; i < ids.size(); i++){
          std::vector<cv::Point2f> tag = corners[i];
          // Compute C.G. of tag
          float cy = 0.0, cz = 0.0;
          std::vector<cv::Point2f> vertices;
          for (cv::Point2f const & p : tag)
          {
              cy += p.x; cz += p.y;
              cv::Point2f pix;
              pix.x = (p.x - 0.5*(float)w)/(0.5*(float)w)*(h/(double)w);
              pix.y = (p.y - 0.5*(float)h)/(0.5*(float)w)*(h/(double)w);
              vertices.push_back(pix);
          }
          // Calculate area of ArUco code
          size_t n = vertices.size();
          int j = n - 1;
          float area = 0;
          for (size_t i = 0; i < n; i++)
          {
              area += (vertices[j].x + vertices[i].x)*(vertices[j].y - vertices[i].y);
              j = i;
          }
          if (n) { cy /= n; cz /= n; area = abs(area/2.0); }

          // Pack message
          ArUco_msg msg;
          msg.py        = (cy - 0.5*(float)w)/(0.5*(float)w)*(h/((double)w));  // or *(IMGHEIGHT/IMGWIDTH)
          msg.pz        = (cz - 0.5*(float)h)/(0.5*(float)w)*(h/((double)w));  // or *(IMGHEIGHT/IMGWIDTH)
          msg.time      = inimg.time_stmp.tv_sec + inimg.time_stmp.tv_usec/1000000.;
          msg.psqrtA    = sqrt(area);

          // Encode message
          msg.ArUco_header.messageSize = sizeof(struct ArUco_msg);
          int byteCount         = msg.ArUco_header.messageSize;
          int headerSize        = sizeof(struct msg_header);
          int index             = headerSize;
          unsigned int csum     = calculateCheckSum((unsigned char *)&msg, byteCount, index);
          unsigned int hcsum    = calculateCheckSum((unsigned char *)&msg, headerSize - sizeof(int)*2, 0);
          msg.ArUco_header.csum = csum;
          msg.ArUco_header.hcsum= hcsum;
          std::string ArUco_string = encodeSerialMsg((char *)&msg, byteCount);
          jevois::Module::sendSerial(ArUco_string);
      }

      // Send the output image with our processing results to the host over USB:
      outframe.send(); // NOTE: optional here, outframe destructor would call it anyway
    }
    // ####################################################################################################
  protected:
    std::shared_ptr<ArUco> itsArUco;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ArUcoDetect);
