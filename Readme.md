# Billiard Ball Tracking with Kalman Filter - DSIP Final Exam Project

## Introduction
This project was done as a part of the final exam of Digital Signal and Image Processing (DSIP) class at University of Genova. I used Kalman Filter to track a billiard ball in 2D. I used video clips which were captured by my colleague Andrea Bricola. There is a folder with all the code, Which has a Kalman.py file which contains the class KF and unit test file test_Kalman.py for sanity check. Lastly there is a main.py file which initialises video clips, using opencv, extracts the coordinates of the ball by finding its contours. And then uses the KF class to predict the next position of the ball. As you can seen the plots, the tracking is lost for a few seconds when the ball hits one of the corners and rebounds.

## Project Overview
- **Data Source**: The video clips used for this project were captured by my colleague, Andrea Bricola, and provided as the input data for the tracking task.
- **Kalman Filter Implementation**: The core algorithm responsible for tracking the billiard ball is implemented in the `Kalman.py` file. It contains the `KF` class, which encapsulates the Kalman Filter logic.
- **Unit Testing**: To ensure the correctness of the Kalman Filter implementation, a unit test file named `test_Kalman.py` was included. The test suite in this file performs sanity checks on the Kalman Filter functions and verifies its accuracy.
- **Main Application**: The `main.py` file serves as the entry point of the application. It initializes the video clips using OpenCV, extracts the coordinates of the billiard ball by detecting its contours, and then employs the Kalman Filter class (`KF`) to predict the next position of the ball.

## Tracking Behavior
As depicted in the generated plots, the tracking performance exhibits a temporary loss of accuracy when the billiard ball hits one of the corners and rebounds. This behavior is expected and arises from the nature of the Kalman Filter, which relies heavily on the consistency of measurements. However, overall, the Kalman Filter successfully predicts the ball's position during most of its motion, providing a reliable tracking mechanism.

## Project Structure
The project directory contains the following files:

1. `Kalman.py`: This file contains the implementation of the `KF` class, which represents the Kalman Filter algorithm used for tracking the billiard ball.

2. `test_Kalman.py`: This file houses the unit tests for the Kalman Filter implementation. It ensures that the Kalman Filter functions perform as expected and produce accurate results.

3. `main.py`: The main application file, responsible for initializing the video clips, extracting the billiard ball's coordinates, and applying the Kalman Filter to predict its future positions.

4. `data/`: A folder containing the video clips captured by Andrea Bricola. These clips serve as the input data for the tracking process.

## Usage
To run the project, ensure you have the required dependencies installed. The project relies on OpenCV for video processing and visualization. After installing the dependencies, execute the `main.py` script. It will process the video clips, track the billiard ball, and generate the necessary plots and visualizations.

## Acknowledgments
I would like to express my gratitude to my colleague, Andrea Bricola, for providing the video clips used in this project. Their contribution was instrumental in the successful completion of the billiard ball tracking task.

## Contact
For any questions or inquiries regarding this project, please feel free to reach out to me at [hannanmustajab[at]icloud[dot]com.

---

Note: The provided information is a summarized version of the project's details. The full code and documentation can be found in the project folder. 