import argparse

def parse_cli_arguments() -> str:
    """
    Extracts the root folder path from terminal arguments, 
    ignoring Streamlit's internal arguments.
    """
    parser = argparse.ArgumentParser(description="cTrader Multi-Folder Optimization Analyzer")
    parser.add_argument(
        "-d",
        "--dir",
        "--directory",
        dest="root_directory",
        type=str,
        default=".",
        help="Path to the root directory containing numbered optimization folders (0, 1, 2...).",
    )
    args, _ = parser.parse_known_args()
    return args.root_directory
