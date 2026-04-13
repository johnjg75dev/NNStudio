"""Entry point — python run.py"""

import argparse
import os
import subprocess
import sys
from app import create_app


def run_server():
    """Run the Flask development server."""
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


def list_tests():
    """List all available test files."""
    tests_dir = "tests"
    if os.path.exists(tests_dir):
        test_files = [
            f
            for f in os.listdir(tests_dir)
            if f.startswith("test_") and f.endswith(".py")
        ]
        print("Available tests:")
        for test in test_files:
            print(f"  - {test[:-3]}")  # Remove .py extension
        print(
            "\nUse: python run.py test <test_name> [<test_name> ...] to run specific tests"
        )
        print("Or: python run.py test all to run all tests")
    else:
        print("Tests directory not found.")


def run_tests(selected_tests, html_report=None):
    """Run specified tests using pytest."""
    if "all" in selected_tests or not selected_tests:
        # Run all tests
        cmd = ["pytest", "tests/"]
    else:
        # Run specific tests with partial, case-insensitive matching
        test_files = []
        tests_dir = "tests"

        # Get all test files
        all_test_files = [
            f
            for f in os.listdir(tests_dir)
            if f.startswith("test_") and f.endswith(".py")
        ]

        for test_input in selected_tests:
            matched = False
            for test_file in all_test_files:
                # Remove .py and check if input is contained in test name (case-insensitive)
                test_name = test_file[:-3]  # remove .py
                if test_input.lower() in test_name.lower():
                    test_files.append(f"tests/{test_file}")
                    matched = True

            if not matched:
                print(f"Warning: No test matches '{test_input}'. Skipping.")

        if test_files:
            cmd = ["pytest"] + test_files
        else:
            print("No valid tests specified.")
            return

    # Add HTML reporting if requested
    if html_report:
        # Insert --html and --self-contained-html options right after pytest
        cmd.insert(1, f"--html={html_report}")
        cmd.insert(2, "--self-contained-html")
        cmd.insert(3, "--tb=long")
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="NNStudio CLI", add_help=True)
    parser.add_argument(
        "command",
        choices=["server", "test"],
        help="Command to execute: server (run Flask server) or test (run tests)",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Test names to run (without .py extension). Use 'all' to run all tests. "
        "When omitted, lists available tests.",
    )
    parser.add_argument(
        "-o",
        "--html",
        dest="html_report",
        help="Generate HTML test report. Provide filename (e.g., 'report.html') or directory (e.g., 'reports/').",
    )

    args = parser.parse_args()

    if args.command == "server":
        run_server()
    elif args.command == "test":
        if not args.tests:
            list_tests()
        else:
            run_tests(args.tests, html_report=args.html_report)


if __name__ == "__main__":
    main()
