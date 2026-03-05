**ROVER RESEARCH TEAM**

APPROVED

OPEN LIBRARIES FOR AUTONOMOUS NAVIGATION OF A SIX-WHEELED PROTOTYPE ROVER ACROSS MODERATELY ROUGH UNKNOWN TERRAIN WITH VISUAL RECOGNITION OF NAVIGATION TARGETS.

**Open library for recognition of directional movement markers and determination of distance and direction to the markers.**

**Application Description.**

**ROVER-CV-V1.0.0**

**(open library on the Internet)**

**2025**

# **ABSTRACT.**

The open library for recognition of directional movement markers and determination of distance and direction to the markers was developed as part of the project “Development of open libraries for autonomous navigation of a six-wheeled prototype rover across moderately rough unknown terrain with visual recognition of navigation targets”. The project was funded through a research and innovation grant provided by a national technology development foundation under a grant agreement dated December 23, 2024.

In this project, fully autonomous navigation (movement) mode refers to a mode in which a rover with Ackermann steering geometry independently, without commands from a human operator, moves across moderately rough and unknown terrain following directional movement markers until reaching the final target marker, can perform pre-programmed actions at each marker, and can independently return to the starting point. The operator may observe video streams and telemetry transmitted from the rover on a monitoring interface.

Directional movement markers are implemented as signs (with a white background and a black arrow) measuring 300 × 200 mm, raised 100–150 mm above the surface. The arrow dimensions are specified in Appendix A of this document. The final target marker is represented by an orange traffic cone with dimensions corresponding to section 4.3.1.1 of a standard road safety specification.

The “Open library for recognition of directional movement markers and determination of distance and direction to the markers” is intended for automatic detection and recognition of directional movement markers and the final target marker, automatic estimation of distance to them, and transmission of information about detected navigation objects to a localization module for rover motion decision making.

The library is implemented in the Python 3 programming language using the YOLOv8 (Ultralytics) neural network model and is designed for the ROS2 Humble platform (Robot Operating System 2, Humble release).

# **CONTENTS.**

1. [ABSTRACT](#abstract)
2. [Purpose of the Program](#purpose-of-the-program)
3. [Conditions of Use](#conditions-of-use)
4. [Problem Description](#problem-description)
5. [Input and Output Data](#input-and-output-data)
6. [APPENDIX A](#appendix-a)

# **Purpose of the Program.**

The “Open library for recognition of directional movement markers and determination of distance and direction to the markers” is intended for automatic detection and recognition of directional movement markers and the final target marker, automatic estimation of distance to them, and transmission of information about detected navigation objects to a localization module for motion decision making for a six-wheeled prototype rover with Ackermann steering geometry moving across moderately rough unknown terrain.

**APPLICATION AREAS.**

This open library can be used on autonomous mobile platforms on Earth or in space for geological and environmental exploration, investigation and development of hard-to-reach territories, monitoring of energy infrastructure (including nuclear facilities) without shutdown, under conditions of poor communication and absence of up-to-date terrain maps. Additional possible applications include:

- agricultural autonomous technologies that use visual markers for navigation, for example during field plowing;

- construction robots capable of operating in continuously changing environments.

**TECHNICAL CHARACTERISTICS.**

Platform: ROS2 Humble (Robot Operating System 2, Humble release).  
Programming language: Python 3.

Neural network model: YOLOv8 (Ultralytics).

Dependencies:

- opencv-python (computer vision);
- numpy (numerical computations);
- ultralytics (YOLO detection);
- ROS2 (robot integration).

Measurements:

- calibrated distance estimation using piecewise linear interpolation from a calibration table;
- angular measurement based on the pinhole camera model.

Operating range:

- distance: 1–10 meters (calibrated range);
- field of view: depends on camera parameters.

**MINIMUM TECHNICAL REQUIREMENTS:**

- Operating system: Ubuntu 22.04.
- Processor: x86_64 or ARM64 (e.g., Nvidia Jetson).
- RAM: minimum 4 GB, 8 GB recommended.
- GPU: NVIDIA GPU with CUDA 11.0+ support (optional but recommended).
- Disk space: 2 GB for the library and models.
- Camera: USB/CSI camera with minimum resolution 640×480.

**Recommended requirements (tested configuration)**

- Platform: Nvidia Jetson Orin NX class system.
- Camera: industrial global-shutter camera or equivalent.
- RAM: 16 GB.
- Storage: NVMe SSD 256 GB.

**Limitations on the application domain.**

Each directional movement marker and final target marker must be located inside an imaginary circle with a radius of 2 meters. During autonomous movement the rover must stop within this circle for at least 10 seconds before proceeding to the next marker. At least half of the rover must remain inside this circle during the stop period. Collisions between the rover and markers are not allowed. The mandatory 10-second pause at each marker allows the autonomous navigation code to be extended with additional pre-programmed actions at each marker (for example soil sampling or atmospheric measurements).

# **Conditions of Use.**

Directional movement markers and the final target marker must be recognized under the following conditions.

1. Maximum detection distance not less than 10 m  
2. Maximum rover roll angle during autonomous motion: not less than 30 degrees  
3. Maximum obstacle height the rover can overcome: 100 mm  
4. Real-time operation during vibration while moving across moderately rough terrain, including conditions of elevated ambient temperature (30–40 °C) and simulated planetary terrain with reduced visibility due to dust.

Directional movement markers are implemented as signs (white background with black arrow) measuring 300 × 200 mm and mounted 100–150 mm above the surface. Arrow dimensions are specified in Appendix A. The final target marker is represented by an orange traffic cone of standardized size.

Operation of the software library for recognition of directional movement markers and determination of distance and direction to the markers is guaranteed when running on platforms equipped with hardware GPUs with near-zero idle power consumption, when using cameras with global shutter sensors, and at rover speeds not exceeding 5 km/h. Testing of the library was conducted on a compact GPU-accelerated robotics platform with an industrial global-shutter camera.

# **Problem Description.**

The goal of detection algorithms is classification and localization of objects in an image. In the context of this project, the goal of the detection algorithm is to identify navigation markers of a strictly defined shape, color, and size, which are then used to determine the distance and direction (turn angle) to the localized navigation marker as well as the direction of motion toward the next waypoint indicated by the marker.

The software library for recognition of directional movement markers and determination of distance and direction to the markers uses an image detection approach based on the YOLOv8n convolutional neural network (YOLO version 8, nano variant). During dataset preparation, the SAM 2.1 (Segment Anything Model 2.1) was used to accelerate object labeling. Automatic segmentation and video object tracking with segmentation were applied, allowing object masks to propagate between frames in video sequences and ensuring annotation consistency.

**Input and Output Data.**

Input to the software library consists of color images from rover cameras in BGR format.

The software library provides a minimal and sufficient set of outputs for transmitting information about detected navigation objects to the localization module:

- object type (arrow/cone);
- movement direction (left/right);
- distance to the object (meters);
- angular position (degrees);
- detection confidence.

# **APPENDIX A.**

Directional movement marker.
![arrow1]