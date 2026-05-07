#!/usr/bin/env python3
"""
Quick Start: Seed the Use-Case Gallery Database

This script demonstrates how to populate the NNStudio database with all
60 use cases across 10 categories. Run this once after deploying the code.

Usage:
  python quick_seed.py
"""

import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from app import create_app
    from scripts.seed_usecases import seed_usecases

    print("=" * 70)
    print("NNStudio Use-Case Gallery — Database Seeding")
    print("=" * 70)
    print()

    # Create Flask app (this initializes DB)
    app = create_app()

    # Seed the database
    with app.app_context():
        try:
            seed_usecases()
            print()
            print("=" * 70)
            print("✓ SUCCESS: Database seeded with 60 use cases!")
            print("=" * 70)
            print()
            print("Next steps:")
            print("  1. Start the app: python run.py")
            print("  2. Visit http://localhost:5000/ to see the gallery")
            print("  3. Admin panel: http://localhost:5000/admin/usecases")
            print()
            print("API endpoints:")
            print("  GET  /api/usecases/gallery          (public)")
            print("  GET  /api/usecases/categories       (public)")
            print("  GET  /api/usecases/admin            (admin)")
            print("  PUT  /api/usecases/admin/<id>       (admin)")
            print()
        except Exception as e:
            print()
            print("=" * 70)
            print("✗ ERROR: Failed to seed database")
            print("=" * 70)
            print()
            print(f"Error: {e}")
            print()
            import traceback
            traceback.print_exc()
            sys.exit(1)
