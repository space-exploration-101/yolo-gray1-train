import sys

import ultralytics


print(f"PYTHON={sys.version.split()[0]}")
print(f"ULTRALYTICS_VERSION={ultralytics.__version__}")
print(f"ULTRALYTICS_FILE={ultralytics.__file__}")
