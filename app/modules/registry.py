"""
app/modules/registry.py
ModuleRegistry — discovers and indexes every BaseModule subclass found
under app/modules/{functions,architectures,presets,optimizers}.

Discovery rules
───────────────
• Each sub-folder is scanned for .py files (excluding __init__ / base).
• Every Python file is imported as a subpackage.
• Any class that:
    – is a subclass of BaseModule
    – is NOT BaseModule itself
    – has a non-empty .key attribute
  … is registered automatically.

No manual registration list required.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Type
from flask import current_app

from .base import BaseModule


def get_registry() -> ModuleRegistry:
    """Helper to retrieve the global registry from the current app."""
    return current_app.extensions.get("module_registry")


class ModuleRegistry:
    # Sub-packages to scan (relative to this file's parent directory)
    _SCAN_PACKAGES = [
        "app.modules.functions",
        "app.modules.architectures",
        "app.modules.presets",
        "app.modules.optimizers",
    ]

    def __init__(self):
        # key → BaseModule *instance*
        self._modules: dict[str, BaseModule] = {}
        # category → list[key]
        self._by_category: dict[str, list[str]] = {}

    # ── discovery ────────────────────────────────────────────────────
    def discover(self):
        """Import every module file in the scan packages and register classes."""
        root = Path(__file__).parent.parent  # app/

        for pkg_name in self._SCAN_PACKAGES:
            # Convert dotted package name → filesystem path
            rel_path = pkg_name.replace(".", "/")
            pkg_dir  = root.parent / rel_path   # project root / app/modules/...

            if not pkg_dir.exists():
                continue

            # Import the package itself first (triggers __init__.py if any)
            try:
                importlib.import_module(pkg_name)
            except ImportError:
                pass

            # Walk all .py files in the folder
            for finder, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
                full_name = f"{pkg_name}.{module_name}"
                try:
                    mod = importlib.import_module(full_name)
                except Exception as exc:
                    print(f"[ModuleRegistry] Failed to import {full_name}: {exc}")
                    continue

                # Inspect every attribute for BaseModule subclasses
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type)
                            and issubclass(attr, BaseModule)
                            and attr is not BaseModule
                            and attr.key):
                        self._register_class(attr)

    def load_architectures_from_database(self):
        """Load built-in architectures from the database."""
        try:
            from flask import current_app
            from ..models import BuiltinArchitecture
            from .architectures.database_architecture import DatabaseArchitecture
            
            # Only attempt if we're in app context and database is set up
            if not current_app:
                return
            
            architectures = BuiltinArchitecture.query.all()
            for arch_record in architectures:
                # Skip if already registered (Python file version takes precedence)
                if arch_record.key in self._modules:
                    continue
                
                # Wrap database record and register it
                arch_module = DatabaseArchitecture(arch_record)
                self._modules[arch_record.key] = arch_module
                cat = "architectures"
                self._by_category.setdefault(cat, [])
                if arch_record.key not in self._by_category[cat]:
                    self._by_category[cat].append(arch_record.key)
        except Exception as exc:
            # Database might not be initialized yet, silently continue
            pass

    def _register_class(self, cls: Type[BaseModule]):
        """Instantiate and register a discovered module class."""
        if cls.key in self._modules:
            return  # already registered (e.g. imported twice)
        instance = cls()
        self._modules[cls.key] = instance
        cat = cls.category or "general"
        self._by_category.setdefault(cat, [])
        if cls.key not in self._by_category[cat]:
            self._by_category[cat].append(cls.key)

    # ── retrieval ────────────────────────────────────────────────────
    def get(self, key: str) -> BaseModule | None:
        return self._modules.get(key)
    
    def get_with_custom(self, key: str, user_id: int | None = None) -> BaseModule | None:
        """
        Get a module by key, checking custom functions if static module not found.
        
        Args:
            key: Module key (e.g., 'custom_5' for custom function with ID 5)
            user_id: Current user ID (required to filter custom functions)
            
        Returns:
            BaseModule instance or None
        """
        # Try static module first
        if key in self._modules:
            return self._modules[key]
        
        # Try custom function if key starts with 'custom_'
        if key.startswith('custom_') and user_id:
            try:
                from ..models import CustomTrainingFunction
                from .functions.custom_function_wrapper import DynamicCustomFunction
                
                custom_id = int(key.split('_')[1])
                custom_func = CustomTrainingFunction.query.filter_by(
                    id=custom_id,
                    user_id=user_id
                ).first()
                
                if custom_func and custom_func.is_valid:
                    return DynamicCustomFunction(custom_func)
            except Exception:
                pass
        
        return None

    def all_of_category(self, category: str) -> list[BaseModule]:
        keys = self._by_category.get(category, [])
        return [self._modules[k] for k in keys if k in self._modules]

    def all(self) -> list[BaseModule]:
        return list(self._modules.values())

    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    def to_dict(self) -> dict:
        """Full registry dump for the frontend."""
        result: dict[str, list[dict]] = {}
        for cat, keys in self._by_category.items():
            result[cat] = [
                self._modules[k].to_dict()
                for k in keys if k in self._modules
            ]
        return result
