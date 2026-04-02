"""
Migration script to populate BuiltinArchitecture table from existing Python files.
Run once to seed the database with existing architectures.

Usage:
    python scripts/migrate_architectures.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import create_app, db
from app.models import BuiltinArchitecture
from app.modules.architectures.mlp import MLPArchitecture
from app.modules.architectures.cnn import CNNArchitecture
from app.modules.architectures.rnn import RNNArchitecture
from app.modules.architectures.transformer import TransformerArchitecture
from app.modules.architectures.autoencoder import AutoencoderArchitecture
from app.modules.architectures.vae import VAEArchitecture
from app.modules.architectures.gan import GANArchitecture
from app.modules.architectures.diffusion import DiffusionArchitecture
from app.modules.architectures.vit import ViTArchitecture


ARCHITECTURES = [
    MLPArchitecture(),
    CNNArchitecture(),
    RNNArchitecture(),
    TransformerArchitecture(),
    AutoencoderArchitecture(),
    VAEArchitecture(),
    GANArchitecture(),
    DiffusionArchitecture(),
    ViTArchitecture(),
]


def migrate_architectures():
    """Populate BuiltinArchitecture table from Python modules."""
    app = create_app()
    
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        created_count = 0
        skipped_count = 0
        
        for arch in ARCHITECTURES:
            # Check if already exists
            existing = BuiltinArchitecture.query.filter_by(key=arch.key).first()
            if existing:
                print(f"⊘ Skipped '{arch.key}' (already exists)")
                skipped_count += 1
                continue
            
            # Create new record
            db_arch = BuiltinArchitecture(
                key=arch.key,
                label=arch.label,
                description=arch.description,
                accent_color=arch.accent_color,
                diagram_type=arch.diagram_type,
                trainable=arch.trainable,
                is_autoencoder=getattr(arch, 'is_autoencoder', False),
                is_migrated=True,
            )
            db.session.add(db_arch)
            print(f"✓ Created '{arch.key}': {arch.label}")
            created_count += 1
        
        db.session.commit()
        print(f"\n✓ Migration complete: {created_count} created, {skipped_count} skipped")


if __name__ == "__main__":
    migrate_architectures()
