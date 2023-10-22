import sqlite3

from dataclasses import dataclass
from pathlib import Path
from invokeai.app.services.config import InvokeAIAppConfig
from shutils import copy
from typing import Optional, Dict, Set
from uuid import uuid4
from .hash import FastModelHash
from ..model_management.model_probe import ModelProbe
from ..model_management import (BaseModelType, ModelType)

# this should be derived from the modeltype enum
MODEL_TYPES = {'vae', 'lora', 'controlnet', 'embedding',
               'ip_adapter', 'clip_vision', 't2i_adapter',
               'text_encoder', 'scheduler', 'tokenizer',
               'unet',
               }
MODEL_SQL_ENUM = ','.join([f'"{x}"' for x in MODEL_TYPES])

BASE_TYPES = {'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner'}
BASE_SQL_ENUM = ','.join([f'"{x}"' for x in BASE_TYPES])

@dataclass
class ModelPart:
    type: ModelType
    path: Path

@dataclass
class ModelConfig:
    name: str
    description: str
    base_models: Set[BaseModelType]
    parts: Dict[str, ModelPart]

class NormalizedModelManager():

    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _blob_directory: Path

    def __init__(self, config=InvokeAIAppConfig):
        database_file = config.db_path.parent / 'normalized_models.db'
        Path(database_file).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(database_file, check_same_thread=True)
        self._conn.isolation_level = 'DEFERRED'
        self._cursor = self._conn.cursor()
        self._blob_directory = config.root_path / 'model_blobs'
        self._blob_directory.mkdir(parents=True, exist_ok=True)

        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()
        self._conn.commit()

    def ingest_simple_model(self, model_path: Path) -> int:
        """Insert a simple one-part model, returning its ID."""
        model_name = model_path.stem
        model_hash = FastModelHash.hash(model_path)

        try:
            # retrieve or create the single part that goes into this model
            part_id = self._lookup_part_by_hash(model_hash) or self._install_part(model_hash, model_path)

            # create the model name/source entry
            self._cursor.execute(
                """--sql
                INSERT INTO model_name (
                   name, source, description, is_pipeline
                )
                VALUES (?, ?, ?, 0);
                """,
                (model_name, model_path, f"Imported model {model_name}"),
            )

            # associate the part with the model
            model_id = self._cursor.lastrowid
            self._cursor.execute(
                """--sql
                INSERT INTO model_parts (
                    model_id, part_id
                )
                VALUES (?, ?);
                """,
                (model_id, part_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e

        return model_id

    def ingest_pipeline_model(self, model_path: Path) -> int:
        pass
        

    # in this p-o-p implementation, we assume that the model name is unique
    def get_model(self, name: str) -> Optional[ModelConfig]:
        self._cursor.execute(
            """--sql
            SELECT a.source, a.description, c.type, b.part_name, b.path, d.base
            FROM model_name as a,
                 model_parts as b,
                 model_part as c,
                 model_base as d,
            WHERE a.model_id=?
              AND a.model_id=b.model_id
              AND b.part_id=c.part_id
              AND b.part_id=d.part_id;
            """,
            (name,),
        )
        rows = self._cursor.fetchall()
        if len(rows) == 0:
            return None

        bases: Set[BaseModelType] = {
            BaseModelType(x['base']) for x in rows
        }
        parts: Dict[str, ModelPart] = {
            x['part_name']: ModelPart(
                type=ModelType(x['type']),
                path=Path(x['path'])
            ) for x in rows
        }

        return ModelConfig(
            name=name,
            description=rows[0]['description'],
            base_models=bases,
            parts=parts
        )

    def _lookup_part_by_hash(self, hash: str) -> Optional[int]:
        self._cursor.execute(
            """--sql
            SELECT part_id from model_part
            WHERE hash=?;
            """,
            (hash,),
        )
        rows = self._cursor.fetchone()
        if not rows:
            return None
        return rows[0]

    # may raise an exception
    def _install_part(self, model_hash: str, model_path: Path) -> int: 
        model_info = ModelProbe.probe(model_path)
        model_type = model_info.model_type
        model_base = model_info.base_type
        model_bases = set()

        # hardcoded logic to test multiple base type compatibility
        if model_type == ModelType('vae') and model_base == BaseModelType('sd-1'):
            model_bases = {'sd-1', 'sd-2'}
        elif model_base == BaseModelType('any'):
            model_bases = BASE_TYPES
        else:
            model_bases = {model_base}

        # make the storage name slightly easier to interpret
        blob_name = model_type.value + '-' + uuid4()
        if model_path.is_file() and model_path.suffix:
            blob_name += model_path.suffix

        destination = self._blob_directory / blob_name
        assert not blob_name.exists(), f"a path named {destination} already exists"
        copy(model_path, destination)

        # create entry in the model_path table
        self._cursor.execute(
            """--sql
            INSERT INTO model_part (
               type, hash, path
            )
            VALUES (?, ?, ?);
            """,
            (model_type.value, model_hash, destination),
        )

        # id of the inserted row
        part_id = self._cursor.lastrowid

        # create base compatibility info
        for base in model_bases:
            self._cursor.execute(
                """--sql
                INSERT INTO model_base (id, base)
                VALUES (?, ?);
                """,
                (part_id, base),
            )

        return part_id

    def _create_tables(self):
        self._cursor.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS model_part (
               part_id INTEGER PRIMARY KEY,
               type TEXT CHECK( type IN ({MODEL_SQL_ENUM}) ) NOT NULL,
               hash TEXT UNIQUE,
               refcount INTEGER NOT NULL DEFAULT '0',
               path TEXT NOT NULL
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_name (
              model_id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              source TEXT,
              description TEXT,
              is_pipeline BOOLEAN NOT NULL DEFAULT '0',
              table_of_contents TEXT, -- this is the contents of model_index.json
              UNIQUE(name)
            );
            """
        )
        self._cursor.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS model_base (
                part_id TEXT NOT NULL,
                base TEXT CHECK( base in ({BASE_SQL_ENUM}) ) NOT NULL,
                FOREIGN KEY(part_id) REFERENCES model_part(part_id),
                unique(id,base)
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_parts (
               model_id INTEGER NOT NULL,
               part_id INTEGER NOT NULL,
               part_name TEXT DEFAULT 'root',
               FOREIGN KEY(model_id) REFERENCES model_name(model_id),
               FOREIGN KEY(part_id) REFERENCES model_part(part_id),
               unique(model_id, part_id)
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS insert_model_refcount
            AFTER INSERT
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE model_part SET refcount=refcount+1 WHERE model_part.part_id=new.part_id;
            END;
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS delete_model_refcount
            AFTER DELETE
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE model_part SET refcount=refcount-1 WHERE model_part.part_id=old.part_id;
            END;
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS update_model_refcount
            AFTER UPDATE
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE model_part SET refcount=refcount-1 WHERE model_part.part_id=old.part_id;
               UPDATE model_part SET refcount=refcount+1 WHERE model_part.part_id=new.part_id;
            END;
            """
        )
