import xml.etree.ElementTree as ET

def load_its(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root
