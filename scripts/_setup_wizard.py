"""
Setup wizard directory structure.
Run once: python -c "from _setup_wizard import setup_wizard_dirs; setup_wizard_dirs()"
"""
import os

def setup_wizard_dirs():
    """Create wizard directory structure."""
    base = "app/wizard"
    dirs = [
        f"{base}",
        f"{base}/steps",
        f"{base}/plugins",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init_file = os.path.join(d, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Wizard module\n")
    
    print(f"✓ Created wizard directory structure: {base}/")

if __name__ == "__main__":
    setup_wizard_dirs()
