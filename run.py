"""Entry point — python run.py"""

import argparse
import os
import subprocess
import sys
from app import create_app


def run_server(host="0.0.0.0", port=5000, debug=True):
    """Run the Flask development server.

    ``debug`` enables the Werkzeug reloader and interactive debugger.  Turn it
    off (``--no-debug``) whenever the port is reachable from outside the machine.
    """
    app = create_app()
    app.run(host=host, port=port, debug=debug)


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
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Interface the server binds to (default: 0.0.0.0, or $HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "5000")),
        help="Port to listen on (default: 5000, or $PORT).",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable the Werkzeug reloader and debugger. Use this when the "
        "server is exposed beyond localhost.",
    )
    parser.set_defaults(debug=True)

    args = parser.parse_args()

    if args.command == "server":
        run_server(host=args.host, port=args.port, debug=args.debug)
    elif args.command == "test":
        if not args.tests:
            list_tests()
        else:
            run_tests(args.tests, html_report=args.html_report)


if __name__ == "__main__":
    main()
