import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import copy, copytree
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline

from invokeai.app.services.config import InvokeAIAppConfig

from ..model_management import BaseModelType, ModelNotFoundException, DuplicateModelException, ModelType, SubModelType
from ..model_management.convert_ckpt_to_diffusers import convert_ckpt_to_diffusers
from ..model_management.model_probe import InvalidModelException, ModelProbe, ModelVariantType
from ..util.devices import choose_torch_device, torch_dtype
from .hash import FastModelHash

# We create a new enumeration for model types
model_types = {x.name: x.value for x in ModelType}
model_types.update({x.name: x.value for x in SubModelType})
ExtendedModelType = Enum("ExtendedModelType", model_types, type=str)

# Turn into a SQL enum
MODEL_TYPES = {x.value for x in ExtendedModelType}
MODEL_SQL_ENUM = ",".join([f'"{x}"' for x in MODEL_TYPES])

# Again
BASE_TYPES = {x.value for x in BaseModelType}
BASE_SQL_ENUM = ",".join([f'"{x}"' for x in BASE_TYPES])


@dataclass
class ModelPart:
    """Type and path of a pipeline submodel."""
    type: ExtendedModelType
    path: Path
    refcount: int


@dataclass
class SimpleModelConfig:
    """Submodel name, description, type and path."""
    name: str
    description: str
    base_models: Set[BaseModelType]
    type: ExtendedModelType
    path: Path


@dataclass
class PipelineConfig:
    """Pipeline model name, description, type and parts."""
    name: str
    description: str
    base_models: Set[BaseModelType]
    parts: Dict[str, ModelPart]


@dataclass
class ModelListing:
    """Slightly simplified object for generating listings."""
    name: str
    description: str
    source: str
    type: ModelType
    base_models: Set[BaseModelType]


class NormalizedModelManager:
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _blob_directory: Path

    def __init__(self, config=InvokeAIAppConfig):
        database_file = config.db_path.parent / "normalized_models.db"
        Path(database_file).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(database_file, check_same_thread=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = "DEFERRED"
        self._cursor = self._conn.cursor()
        self._blob_directory = config.root_path / "model_blobs"
        self._blob_directory.mkdir(parents=True, exist_ok=True)

        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()
        self._conn.commit()

    def ingest(self, model_path: Path, name: Optional[str] = None) -> Union[SimpleModelConfig, PipelineConfig]:
        """Ingest a simple or pipeline model into the normalized models database."""
        model_path = model_path.absolute()
        info = ModelProbe.probe(model_path)
        if info.model_type == ModelType.Main:
            return self.ingest_pipeline_model(model_path, name)
        else:
            return self.ingest_simple_model(model_path, name)

    def ingest_simple_model(self, model_path: Path, name: Optional[str] = None) -> SimpleModelConfig:
        """Insert a simple one-part model, returning its config."""
        model_name = name or model_path.stem
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
                (model_name, model_path.as_posix(), f"Imported model {model_name}"),
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
                (
                    model_id,
                    part_id,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            if isinstance(e, sqlite3.IntegrityError):
                raise DuplicateModelException(f'a model named {model_name} is already in the database') from e
            else:
                raise e

        return self.get_model(model_name)

    def ingest_pipeline_model(self, model_path: Path, name: Optional[str] = None) -> PipelineConfig:
        """Insert the components of a diffusers pipeline."""
        if model_path.is_file():  # convert to diffusers before ingesting
            name = name or model_path.stem
            with TemporaryDirectory() as tmp_dir:
                _convert_ckpt(model_path, Path(tmp_dir))
                result = self._ingest_pipeline_model(Path(tmp_dir), name, source=model_path)
            return result
        else:
            return self._ingest_pipeline_model(model_path, name)

    def _ingest_pipeline_model(
        self, model_path: Path, name: Optional[str] = None, source: Optional[Path] = None
    ) -> PipelineConfig:
        """Insert the components of a diffusers pipeline."""
        model_name = name or model_path.stem

        model_index = model_path / "model_index.json"
        assert (
            model_index.exists()
        ), f"{model_path} does not look like a diffusers model: model_index.json is missing"  # check that it is a diffuers
        with open(model_index, "r") as file:
            toc = file.read()
        base_type = ModelProbe.probe(model_path).base_type
        source = source or model_path

        try:
            # create a name entry for the pipeline and insert its table of contents
            self._cursor.execute(
                """--sql
                INSERT INTO model_name (
                   name, source, description, is_pipeline, table_of_contents
                )
                VALUES(?, ?, ?, "1", ?);
                """,
                (model_name, source.as_posix(), f"Normalized pipeline {model_name}", toc),
            )
            pipeline_id = self._cursor.lastrowid

            # now we create or retrieve each of the parts
            subdirectories = [x for x in model_path.iterdir() if x.is_dir()]
            parts_to_insert = []
            bases_to_insert = []
            for submodel in subdirectories:
                part_name = submodel.stem
                part_path = submodel
                part_hash = FastModelHash.hash(part_path)
                part_id = self._lookup_part_by_hash(part_hash) or self._install_part(part_hash, part_path, {base_type})
                parts_to_insert.append((pipeline_id, part_id, part_name))
                bases_to_insert.append((part_id, base_type.value))
            # insert the parts into the part list
            self._cursor.executemany(
                """--sql
                INSERT INTO model_parts (
                   model_id, part_id, part_name
                )
                VALUES(?, ?, ?);
                """,
                parts_to_insert,
            )
            # update the base types - over time each simple model will get tagged
            # with all the base types of any pipelines that use it, which is a feature... I think?
            self._cursor.executemany(
                """--sql
                INSERT OR IGNORE INTO model_base (
                   part_id, base
                )
                VALUES(?, ?);
                """,
                bases_to_insert,
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            if isinstance(e, sqlite3.IntegrityError):
                raise DuplicateModelException(f'a model named {model_name} is already in the database') from e
            else:
                raise e

        return self.get_pipeline(model_name)

    # in this p-o-p implementation, we assume that the model name is unique
    def get_model(self, name: str, part: Optional[str] = "root") -> SimpleModelConfig:
        """Fetch a simple model. Use optional `part` to specify the diffusers subfolder."""
        self._cursor.execute(
            """--sql
            SELECT a.source, a.description, c.type, b.part_name, c.path, d.base
            FROM model_name as a,
                 model_parts as b,
                 simple_model as c,
                 model_base as d
            WHERE a.name=?
              AND a.model_id=b.model_id
              AND b.part_id=c.part_id
              AND b.part_id=d.part_id
              AND b.part_name=?
            """,
            (name, part),
        )
        rows = self._cursor.fetchall()
        if len(rows) == 0:
            raise ModelNotFoundException

        bases: Set[BaseModelType] = {BaseModelType(x["base"]) for x in rows}

        return SimpleModelConfig(
            name=name,
            description=rows[0]["description"],
            base_models=bases,
            type=ExtendedModelType(rows[0]["type"]),
            path=Path(rows[0]["path"]),
        )

    # in this p-o-p implementation, we assume that the model name is unique
    def get_pipeline(self, name: str) -> PipelineConfig:
        """Fetch a pipeline model."""
        self._cursor.execute(
            """--sql
            SELECT a.source, a.description, c.type, b.part_name, c.path, d.base, c.refcount
            FROM model_name as a,
                 model_parts as b,
                 simple_model as c,
                 model_base as d
            WHERE a.name=?
              AND a.model_id=b.model_id
              AND b.part_id=c.part_id
              AND b.part_id=d.part_id
            """,
            (name,),
        )
        rows = self._cursor.fetchall()
        if len(rows) == 0:
            raise ModelNotFoundException

        # Find the intersection of base models supported by each part.
        # Need a more pythonic way of doing this!
        bases: Dict[str, Set] = dict()
        base_union: Set[BaseModelType] = set()
        parts = dict()
        for row in rows:
            part_name = row["part_name"]
            base = row["base"]
            if not bases.get(part_name):
                bases[part_name] = set()
            bases[part_name].add(base)
            base_union.add(base)
            parts[part_name] = ModelPart(row["type"], row["path"], row["refcount"])
        for base_set in bases.values():
            base_union = base_union.intersection(base_set)

        return PipelineConfig(
            name=name,
            description=rows[0]["description"],
            base_models={BaseModelType(x) for x in base_union},
            parts=parts,
        )

    def list_models(self) -> List[ModelListing]:
        """Get a listing of models. No filtering implemented yet."""
        # get simple models first
        self._cursor.execute(
            """--sql
            SELECT name, source, is_pipeline
            FROM model_name;
            """,
            (),
        )
        results: List[ModelListing] = []
        for row in self._cursor.fetchall():
            if row["is_pipeline"]:
                pipeline = self.get_pipeline(row["name"])
                results.append(
                    ModelListing(
                        name=pipeline.name,
                        description=pipeline.description,
                        source=row["source"],
                        type=ModelType.Main,
                        base_models=pipeline.base_models,
                    )
                )
            else:
                model = self.get_model(row["name"])
                results.append(
                    ModelListing(
                        name=model.name,
                        description=model.description,
                        source=row["source"],
                        type=model.type,
                        base_models=model.base_models,
                    )
                )
        return results

    def export_pipeline(self, name: str, destination: Path) -> Path:
        """Reconstruction the pipeline as a set of symbolic links in folder indicated by destination."""

        # get the model_index.json file (the "toc")
        self._cursor.execute(
            """--sql
            SELECT table_of_contents, is_pipeline
            FROM model_name
            WHERE name=?
            """,
            (name,),
        )
        row = self._cursor.fetchone()
        if row is None:
            raise ModelNotFoundException

        # if the destination exists and is a directory, then we create
        # a new subdirectory using the model name
        if destination.exists() and destination.is_dir():
            destination = destination / name

        # now check that the (possibly new) destination doesn't already exist
        if destination.exists():
            raise OSError(f"{destination}: path or directory exists; won't overwrite")

        if row['is_pipeline']:
            # write the toc
            toc = row[0]
            destination.mkdir(parents=True)
            with open(destination / "model_index.json", "w") as model_index:
                model_index.write(toc)

            # symlink the subfolders
            model = self.get_pipeline(name)
            for part_name, part_config in model.parts.items():
                source_path = destination / part_name
                target_path = part_config.path
                source_path.symlink_to(target_path)
        else:
            model = self.get_model(name)
            destination = Path(destination.as_posix() + model.path.suffix)
            destination.symlink_to(model.path)

        return destination

    def _lookup_part_by_hash(self, hash: str) -> Optional[int]:
        self._cursor.execute(
            """--sql
            SELECT part_id from simple_model
            WHERE hash=?;
            """,
            (hash,),
        )
        rows = self._cursor.fetchone()
        if not rows:
            return None
        return rows[0]

    # may raise an exception
    def _install_part(self, model_hash: str, model_path: Path, base_types: Set[BaseModelType] = set()) -> int:
        (model_type, model_base) = self._probe_model(model_path)
        if model_base is None:
            model_bases = base_types
        else:
            # hack logic to test multiple base type compatibility
            model_bases = set()
            if model_type == ExtendedModelType("vae") and model_base == BaseModelType("sd-1"):
                model_bases = {BaseModelType("sd-1"), BaseModelType("sd-2")}
            elif model_base == BaseModelType("any"):
                model_bases = {BaseModelType(x) for x in BASE_TYPES}
            else:
                model_bases = {BaseModelType(model_base)}

        # make the storage name slightly easier to interpret
        blob_name = model_type.value + "-" + str(uuid4())
        if model_path.is_file() and model_path.suffix:
            blob_name += model_path.suffix

        destination = self._blob_directory / blob_name
        assert not destination.exists(), f"a path named {destination} already exists"

        if model_path.is_dir():
            copytree(model_path, destination)
        else:
            copy(model_path, destination)

        # create entry in the model_path table
        self._cursor.execute(
            """--sql
            INSERT INTO simple_model (
               type, hash, path
            )
            VALUES (?, ?, ?);
            """,
            (model_type.value, model_hash, destination.as_posix()),
        )

        # id of the inserted row
        part_id = self._cursor.lastrowid

        # create base compatibility info
        for base in model_bases:
            self._cursor.execute(
                """--sql
                INSERT INTO model_base (part_id, base)
                VALUES (?, ?);
                """,
                (part_id, BaseModelType(base).value),
            )

        return part_id

    def _create_tables(self):
        self._cursor.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS simple_model (
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
                FOREIGN KEY(part_id) REFERENCES simple_model(part_id),
                UNIQUE(part_id,base)
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_parts (
               model_id INTEGER NOT NULL,
               part_id INTEGER NOT NULL,
               part_name TEXT DEFAULT 'root',  -- to do: use enum
               FOREIGN KEY(model_id) REFERENCES model_name(model_id),
               FOREIGN KEY(part_id) REFERENCES simple_model(part_id),
               UNIQUE(model_id, part_id, part_name)
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS insert_model_refcount
            AFTER INSERT
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE simple_model SET refcount=refcount+1 WHERE simple_model.part_id=new.part_id;
            END;
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS delete_model_refcount
            AFTER DELETE
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE simple_model SET refcount=refcount-1 WHERE simple_model.part_id=old.part_id;
            END;
            """
        )
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS update_model_refcount
            AFTER UPDATE
            ON model_parts FOR EACH ROW
            BEGIN
               UPDATE simple_model SET refcount=refcount-1 WHERE simple_model.part_id=old.part_id;
               UPDATE simple_model SET refcount=refcount+1 WHERE simple_model.part_id=new.part_id;
            END;
            """
        )

    def _probe_model(self, model_path: Path) -> Tuple[ExtendedModelType, Optional[BaseModelType]]:
        try:
            model_info = ModelProbe.probe(model_path)
            return (model_info.model_type, model_info.base_type)
        except InvalidModelException:
            return (ExtendedModelType(model_path.stem), None)


# Adapted from invokeai/backend/model_management/models/stable_diffusion.py
# This code should be moved into its own module
def _convert_ckpt(checkpoint_path: Path, output_path: Path) -> Path:
    """
    Convert checkpoint model to diffusers format.

    The converted model will be stored atat output_path.
    """
    app_config = InvokeAIAppConfig.get_config()

    weights = checkpoint_path
    model_info = ModelProbe.probe(checkpoint_path)
    base_type = model_info.base_type
    variant = model_info.variant_type
    pipeline_class = StableDiffusionInpaintPipeline if variant == "inpaint" else StableDiffusionPipeline
    config_file = app_config.legacy_conf_path / __select_ckpt_config(base_type, variant)
    precision = torch_dtype(choose_torch_device())

    model_base_to_model_type = {
        BaseModelType.StableDiffusion1: "FrozenCLIPEmbedder",
        BaseModelType.StableDiffusion2: "FrozenOpenCLIPEmbedder",
        BaseModelType.StableDiffusionXL: "SDXL",
        BaseModelType.StableDiffusionXLRefiner: "SDXL-Refiner",
    }
    convert_ckpt_to_diffusers(
        weights.as_posix(),
        output_path.as_posix(),
        model_type=model_base_to_model_type[base_type],
        model_version=base_type,
        model_variant=variant,
        original_config_file=config_file,
        extract_ema=True,
        scan_needed=True,
        pipeline_class=pipeline_class,
        from_safetensors=weights.suffix == ".safetensors",
        precision=precision,
    )
    return output_path


def __select_ckpt_config(version: BaseModelType, variant: ModelVariantType):
    ckpt_configs: Dict[BaseModelType, Dict[ModelVariantType, Optional[str]]] = {
        BaseModelType.StableDiffusion1: {
            ModelVariantType.Normal: "v1-inference.yaml",
            ModelVariantType.Inpaint: "v1-inpainting-inference.yaml",
        },
        BaseModelType.StableDiffusion2: {
            ModelVariantType.Normal: "v2-inference-v.yaml",  # best guess, as we can't differentiate with base(512)
            ModelVariantType.Inpaint: "v2-inpainting-inference.yaml",
            ModelVariantType.Depth: "v2-midas-inference.yaml",
        },
        BaseModelType.StableDiffusionXL: {
            ModelVariantType.Normal: "sd_xl_base.yaml",
            ModelVariantType.Inpaint: None,
            ModelVariantType.Depth: None,
        },
        BaseModelType.StableDiffusionXLRefiner: {
            ModelVariantType.Normal: "sd_xl_refiner.yaml",
            ModelVariantType.Inpaint: None,
            ModelVariantType.Depth: None,
        },
    }
    return ckpt_configs[version][variant]
