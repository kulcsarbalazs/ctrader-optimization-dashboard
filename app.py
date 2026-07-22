import sys
from pathlib import Path

# Ensure the project root is in the python path
sys.path.append(str(Path(__file__).parent))

from src.utils import parse_cli_arguments
from src.dashboard import OptimizationDashboard

if __name__ == "__main__":
    cli_root_directory = parse_cli_arguments()
    app = OptimizationDashboard(initial_root_dir=cli_root_directory)
    app.run()
