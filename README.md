# OPEN LIBRARIES FOR AUTONOMOUS NAVIGATION OF A SIX-WHEELED ROVER PROTOTYPE IN MODERATELY ROUGH UNKNOWN TERRAIN WITH VISUAL NAVIGATION TARGET RECOGNITION

## Open Library for Detection of Direction Indicators, Distance Estimation, and Direction Determination

The “Open library for detection of direction indicators, distance estimation, and direction determination” is designed for automatic detection and recognition of direction indicators and the final navigation target, automatic estimation of the distance to them, and transmission of information about detected navigation objects to a localization module for decision-making during the movement of a six-wheeled rover prototype with Ackermann steering geometry in moderately rough unknown terrain.

## Licensing (License)

This software product is distributed under a dual licensing model (This software is dual-licensed):

- MIT License (see [LICENSE](LICENSE))
- Russian Open License (see [GENERAL_LICENSE-RF.md](GENERAL_LICENSE-RF.md))

You may use this project under the terms of either license.

##

The library can be used on autonomous mobile platforms operating on Earth and in space for geological and environmental exploration, research and development of hard-to-reach territories, and monitoring of energy infrastructure (including nuclear power plants) without interrupting operation in environments with poor communication and absence of up-to-date terrain maps.

A directional movement indicator is implemented as a sign (with a white background and a black arrow), measuring 300 × 200 mm and mounted at a height of 100–150 mm above the surface. The arrow dimensions on the sign are specified in the documentation. The final navigation target indicator is an orange traffic cone with dimensions compliant with section 4.3.1.1 of GOST32758-2014.

The software library for recognition of direction indicators and estimation of distance and direction uses an image detection approach based on the convolutional neural network YOLOv8n (YOLO version 8, nano modification). During dataset preparation, the SAM 2.1 (Segment Anything Model 2.1) model was used to accelerate the object annotation process. Automatic segmentation and object tracking in video sequences were employed, which allowed object masks to be propagated between frames and ensured annotation consistency.

<a id="minsystemrequirements"></a>
**Minimum system requirements:**
*  Operating system: Ubuntu 22.04.
*  Processor: x86_64 or ARM64 (for example, Nvidia Jetson).
*  RAM: 4 GB minimum, 8 GB recommended.
*  GPU: NVIDIA GPU with CUDA 11.0+ support (optional but recommended).
*  Disk space: 2 GB for the library and models.
*  Camera: USB/CSI camera with resolution at least 640x480.

**Recommended requirements (tested configuration)**
* Platform: embedded AI computing platform.
* Camera: industrial machine-vision camera or equivalent.
* RAM: 16 GB.
* Storage: NVMe SSD 256 GB.

**Key module files:** 
*   Python3 module `eureka_nav_lib.py`

## 📖 Documentation

Full technical documentation is available in the [docs/](./docs/) directory:

*   [📄 Application description in md format](./docs/Description-CV.md) — application description.
*   [📄 Programmer manual in md format](./docs/Manual-CV.md) — programmer manual for the `eureka_nav_lib.py` library.<br>
&nbsp; <!-- -->
*   [📄 Application description in pdf format](./docs/Description-CV.pdf) — application description.
*   [📄 Programmer manual in pdf format](./docs/Manual-CV.pdf) — programmer manual for the `eureka_nav_lib.py` library.

## Operating Range

*   distance: 1–10 meters (calibrated range);
*   field of view: depends on camera parameters.

Testing of this library was conducted in field conditions on a real rover platform equipped with an embedded AI computing system and an industrial camera.