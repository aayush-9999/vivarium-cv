# pipeline/annotator/factory.py
from pipeline.annotator.opencv_annotator import OpenCVAnnotator

def get_annotator():
    return OpenCVAnnotator()