"""Every SQL statement of the application lives in the database layer.

Everything else reaches the database through `Database.queries`, so a backend port, a schema change or a
query fix happens in one place (`invokeai/app/services/shared/database/`). This test fails when code
outside the database layer imports a database driver, SQLAlchemy or Alembic, executes a statement,
carries SQL text, or reaches into a database object's internals.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("invokeai", "scripts")
DATABASE_LAYER = (
    "invokeai/app/services/shared/database/",
    # The 63 migrations up to the portable cutover are SQLite DDL and stay as written.
    "invokeai/app/services/shared/sqlite_migrator/",
)

# Code that held SQL before the database layer existed, with how many violations each file still has.
# Porting a domain lowers or removes its entries, and a file that ends up with fewer violations than
# allowed fails the test until its number is lowered, so the amount of SQL outside the layer only shrinks.
# Do not raise a number or add a file.
NOT_YET_PORTED: dict[str, int] = {
    "invokeai/app/services/fonts/fonts_default.py": 32,
    "invokeai/app/services/gallery/gallery_default.py": 16,
    "invokeai/app/services/image_index/image_index_records_sqlite.py": 26,
    "invokeai/app/services/image_moves/image_moves_default.py": 58,
    "invokeai/app/services/session_queue/session_queue_sqlite.py": 166,
    # The transitional cursor facade, removed once every service is ported.
    "invokeai/app/services/shared/sqlite/sqlite_database.py": 1,
    "invokeai/backend/util/gallery_maintenance.py": 8,
    "invokeai/frontend/install/import_images.py": 13,
    "scripts/remove_orphaned_models.py": 3,
}

_DRIVER_MODULES = ("sqlite3", "sqlalchemy", "alembic", "pymysql")
_EXECUTE_METHODS = {"execute", "executemany", "executescript", "exec_driver_sql"}
# Internals of the database layer that only it may import.
_LAYER_PACKAGE = "invokeai.app.services.shared.database"
_LAYER_INTERNAL_MODULES = ("engines", "schema", "dialect")
_LAYER_INTERNALS = tuple(f"{_LAYER_PACKAGE}.{module}" for module in _LAYER_INTERNAL_MODULES)
_SQL_STATEMENT = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|PRAGMA|WITH|REPLACE|VACUUM|BEGIN)\b[\s\S]*"
    r"\b(FROM|INTO|SET|TABLE|INDEX|TRIGGER|AS|IMMEDIATE|TRANSACTION|WHERE|VALUES)\b"
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return docstrings


def _is_driver_module(name: str) -> bool:
    return any(name == module or name.startswith(f"{module}.") for module in _DRIVER_MODULES)


def _looks_like_sql(text: str) -> bool:
    return "--sql" in text or _SQL_STATEMENT.match(text) is not None


def scan(path: Path, relative: str) -> list[Violation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    docstrings = _docstring_nodes(tree)
    f_string_parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
    violations: list[Violation] = []

    def add(node: ast.AST, rule: str, detail: str) -> None:
        violations.append(Violation(relative, getattr(node, "lineno", 0), rule, detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_driver_module(alias.name):
                    add(node, "imports a database driver", alias.name)
                elif alias.name.startswith(_LAYER_INTERNALS):
                    add(node, "imports database layer internals", alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            if _is_driver_module(node.module):
                add(node, "imports a database driver", node.module)
            elif node.module.startswith(_LAYER_INTERNALS) or (
                node.module == _LAYER_PACKAGE and any(alias.name in _LAYER_INTERNAL_MODULES for alias in node.names)
            ):
                add(node, "imports database layer internals", node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _EXECUTE_METHODS:
                add(node, "executes SQL", f".{node.func.attr}()")
        elif isinstance(node, ast.Attribute):
            owner_is_self = isinstance(node.value, ast.Name) and node.value.id == "self"
            if node.attr in ("_db", "_conn") and not owner_is_self:
                add(node, "reaches into a database object", f".{node.attr}")
            elif node.attr == "engine" and isinstance(node.ctx, ast.Load) and not owner_is_self:
                add(node, "uses the SQLAlchemy engine", ".engine")
        elif isinstance(node, ast.JoinedStr):
            # An f-string's constant parts are checked together, with the interpolations as placeholders.
            text = "".join(
                part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                for part in node.values
            )
            if _looks_like_sql(text):
                add(node, "carries SQL text", " ".join(text.split())[:60])
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in f_string_parts
            and _looks_like_sql(node.value)
        ):
            add(node, "carries SQL text", " ".join(node.value.split())[:60])
    return violations


def _scanned_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if "node_modules/" in relative or "/__pycache__/" in relative:
                continue
            if relative.startswith(DATABASE_LAYER):
                continue
            files.append((path, relative))
    return files


def test_only_the_database_layer_touches_the_database() -> None:
    offending: list[str] = []
    for path, relative in _scanned_files():
        violations = scan(path, relative)
        allowed = NOT_YET_PORTED.get(relative, 0)
        if len(violations) > allowed:
            offending.append(f"{relative}: {len(violations)} violations, {allowed} allowed")
            offending.extend(f"  {violation}" for violation in violations)
    assert not offending, "Database access outside the database layer; add a query module instead:\n" + "\n".join(
        offending
    )


def test_the_not_yet_ported_allowance_only_shrinks() -> None:
    scanned = {relative: path for path, relative in _scanned_files()}
    stale: list[str] = []
    for relative, allowed in NOT_YET_PORTED.items():
        remaining = len(scan(scanned[relative], relative)) if relative in scanned else 0
        if remaining < allowed:
            action = "remove the entry" if remaining == 0 else f"lower it to {remaining}"
            stale.append(f"{relative}: {remaining} violations left, {allowed} allowed -- {action}")
    assert not stale, "NOT_YET_PORTED allows more than is left:\n" + "\n".join(stale)
