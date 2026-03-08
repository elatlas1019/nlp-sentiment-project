import os
import sys

# Add the project root to sys.path to allow imports from frontend
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from frontend.app import main

if __name__ == "__main__":
    main()
