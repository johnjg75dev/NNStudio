# Architecture Database Migration Guide

## Overview

This migration moves built-in architecture metadata from Python files into the database, allowing admin users to manage architectures without modifying code.

## Files Modified

### 1. **New Database Model** (`app/models/architecture.py`)
- Created `BuiltinArchitecture` model for storing global architecture metadata
- Fields: `key`, `label`, `description`, `accent_color`, `diagram_type`, `trainable`, `is_autoencoder`
- Separate from user-specific `ArchitectureDefinition` model

### 2. **Registry Updates** (`app/modules/registry.py`)
- Added `load_architectures_from_database()` method
- Loads BuiltinArchitecture records and wraps them as modules
- Python file versions take precedence over database (backward compatibility)

### 3. **Dynamic Module Wrapper** (`app/modules/architectures/database_architecture.py`)
- New `DatabaseArchitecture` class wraps database records
- Acts like a normal module so it integrates seamlessly with existing code

### 4. **App Initialization** (`app/__init__.py`)
- Added automatic loading of architectures from database after discovery
- Uses app context to ensure database is initialized

### 5. **Models Export** (`app/models/__init__.py`)
- Exported `BuiltinArchitecture` for use throughout the app

## Migration Steps

### Step 1: Generate Database Tables

Run the migration script to create tables and populate with existing architectures:

```bash
python scripts/migrate_architectures.py
```

Expected output:
```
✓ Created 'mlp': MLP — Fully Connected
✓ Created 'cnn': CNN — Convolutional
✓ Created 'rnn': RNN — Recurrent
✓ Created 'transformer': Transformer
✓ Created 'autoencoder': Autoencoder
✓ Created 'vae': Variational Autoencoder
✓ Created 'gan': GAN — Generative Adversarial
✓ Created 'diffusion': Diffusion
✓ Created 'vit': Vision Transformer

✓ Migration complete: 9 created, 0 skipped
```

### Step 2: Verify

1. Start the app: `python run.py`
2. Check the UI - architectures should load normally
3. You can now edit architecture metadata in the database

## Future Steps

After successful migration, you can:

1. **Optional: Remove Python files** (after confirming database version works)
   - Delete `app/modules/architectures/*.py` files
   - Keep only `base_architecture.py` and `database_architecture.py`

2. **Create Admin Dashboard** to edit architectures directly in the UI
   - Add admin routes to list/update `BuiltinArchitecture` records
   - No code deployment needed for changes

3. **Apply Similar Pattern to Other Modules**
   - `app/modules/functions/` - for custom training functions metadata
   - `app/modules/optimizers/` - for optimizer descriptions
   - `app/modules/presets/` - already done!

## Backward Compatibility

- Python file versions take precedence (if you keep them)
- Database versions only load if Python file doesn't exist for that key
- Can gradually migrate without breaking anything

## Rollback

If needed, simply:
1. Delete all `BuiltinArchitecture` records from the database
2. The registry will fall back to loading from Python files
3. No code changes needed
