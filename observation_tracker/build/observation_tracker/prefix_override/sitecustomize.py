import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/scv/SCV/src/perception/observation_tracker/install/observation_tracker'
