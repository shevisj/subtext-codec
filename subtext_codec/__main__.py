import os
import sys
from .cli import main

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main(sys.argv)
