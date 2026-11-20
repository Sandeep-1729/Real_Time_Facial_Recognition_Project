import sys; print('Python:',sys.version)
import cv2,numpy as np; print('OpenCV:',cv2.__version__,'LBPH available:',hasattr(cv2,'face')); print('NumPy:',np.__version__)
import tensorflow as tf; print('TensorFlow:',tf.__version__)
