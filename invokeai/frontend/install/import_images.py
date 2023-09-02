# Copyright (c) 2023 - The InvokeAI Team
# Primary Author: David Lovell (github @f412design, discord @techjedi)
# co-author, minor tweaks - Lincoln Stein

# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
"""Script to import images into the new database system for 3.0.0"""

import os
import datetime
import shutil
import locale
import sqlite3
import json
import glob
import re
import uuid
import yaml
import PIL
import PIL.ImageOps
import PIL.PngImagePlugin

from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.key_binding import KeyBindings

from invokeai.app.services.config import InvokeAIAppConfig

app_config = InvokeAIAppConfig.get_config()

bindings = KeyBindings()


@bindings.add("c-c")
def _(event):
    raise KeyboardInterrupt


# release notes
# "Use All" with size dimensions not selectable in the UI will not load dimensions


class Config:
    """Configuration loader."""

    def __init__(self):
        pass

    TIMESTAMP_STRING = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    INVOKE_DIRNAME = "invokeai"
    YAML_FILENAME = "invokeai.yaml"
    DATABASE_FILENAME = "invokeai.db"

    database_path = None
    database_backup_dir = None
    outputs_path = None
    thumbnail_path = None

    def find_and_load(self):
        """Find the yaml config file and load"""
        root = app_config.root_path
        if not self.confirm_and_load(os.path.abspath(root)):
            print("\r\nSpecify custom database and outputs paths:")
            self.confirm_and_load_from_user()

        self.database_backup_dir = os.path.join(os.path.dirname(self.database_path), "backup")
        self.thumbnail_path = os.path.join(self.outputs_path, "thumbnails")

    def confirm_and_load(self, invoke_root):
        """Validate a yaml path exists, confirms the user wants to use it and loads config."""
        yaml_path = os.path.join(invoke_root, self.YAML_FILENAME)
        if os.path.exists(yaml_path):
            db_dir, outdir = self.load_paths_from_yaml(yaml_path)
            if os.path.isabs(db_dir):
                database_path = os.path.join(db_dir, self.DATABASE_FILENAME)
            else:
                database_path = os.path.join(invoke_root, db_dir, self.DATABASE_FILENAME)

            if os.path.isabs(outdir):
                outputs_path = os.path.join(outdir, "images")
            else:
                outputs_path = os.path.join(invoke_root, outdir, "images")

            db_exists = os.path.exists(database_path)
            outdir_exists = os.path.exists(outputs_path)

            text = f"Found {self.YAML_FILENAME} file at {yaml_path}:"
            text += f"\n  Database : {database_path}"
            text += f"\n  Outputs  : {outputs_path}"
            text += "\n\nUse these paths for import (yes) or choose different ones (no) [Yn]: "

            if db_exists and outdir_exists:
                if (prompt(text).strip() or "Y").upper().startswith("Y"):
                    self.database_path = database_path
                    self.outputs_path = outputs_path
                    return True
                else:
                    return False
            else:
                print("  Invalid: One or more paths in this config did not exist and cannot be used.")

        else:
            message_dialog(
                title="Path not found",
                text=f"Auto-discovery of configuration failed! Could not find ({yaml_path}), Custom paths can be specified.",
            ).run()
            return False

    def confirm_and_load_from_user(self):
        default = ""
        while True:
            database_path = os.path.expanduser(
                prompt(
                    "Database: Specify absolute path to the database to import into: ",
                    completer=PathCompleter(
                        expanduser=True, file_filter=lambda x: Path(x).is_dir() or x.endswith((".db"))
                    ),
                    default=default,
                )
            )
            if database_path.endswith(".db") and os.path.isabs(database_path) and os.path.exists(database_path):
                break
            default = database_path + "/" if Path(database_path).is_dir() else database_path

        default = ""
        while True:
            outputs_path = os.path.expanduser(
                prompt(
                    "Outputs: Specify absolute path to outputs/images directory to import into: ",
                    completer=PathCompleter(expanduser=True, only_directories=True),
                    default=default,
                )
            )

            if outputs_path.endswith("images") and os.path.isabs(outputs_path) and os.path.exists(outputs_path):
                break
            default = outputs_path + "/" if Path(outputs_path).is_dir() else outputs_path

        self.database_path = database_path
        self.outputs_path = outputs_path

        return

    def load_paths_from_yaml(self, yaml_path):
        """Load an Invoke AI yaml file and get the database and outputs paths."""
        try:
            with open(yaml_path, "rt", encoding=locale.getpreferredencoding()) as file:
                yamlinfo = yaml.safe_load(file)
                db_dir = yamlinfo.get("InvokeAI", {}).get("Paths", {}).get("db_dir", None)
                outdir = yamlinfo.get("InvokeAI", {}).get("Paths", {}).get("outdir", None)
                return db_dir, outdir
        except Exception:
            print(f"Failed to load paths from yaml file! {yaml_path}!")
            return None, None


class ImportStats:
    """DTO for tracking work progress."""

    def __init__(self):
        pass

    time_start = datetime.datetime.utcnow()
    count_source_files = 0
    count_skipped_file_exists = 0
    count_skipped_db_exists = 0
    count_imported = 0
    count_imported_by_version = {}
    count_file_errors = 0

    @staticmethod
    def get_elapsed_time_string():
        """Get a friendly time string for the  time elapsed since processing start."""
        time_now = datetime.datetime.utcnow()
        total_seconds = (time_now - ImportStats.time_start).total_seconds()
        hours = int((total_seconds) / 3600)
        minutes = int(((total_seconds) % 3600) / 60)
        seconds = total_seconds % 60
        out_str = f"{hours} hour(s) -" if hours > 0 else ""
        out_str += f"{minutes} minute(s) -" if minutes > 0 else ""
        out_str += f"{seconds:.2f} second(s)"
        return out_str


class InvokeAIMetadata:
    """DTO for core Invoke AI generation properties parsed from metadata."""

    def __init__(self):
        pass

    def __str__(self):
        formatted_str = f"{self.generation_mode}~{self.steps}~{self.cfg_scale}~{self.model_name}~{self.scheduler}~{self.seed}~{self.width}~{self.height}~{self.rand_device}~{self.strength}~{self.init_image}"
        formatted_str += f"\r\npositive_prompt: {self.positive_prompt}"
        formatted_str += f"\r\nnegative_prompt: {self.negative_prompt}"
        return formatted_str

    generation_mode = None
    steps = None
    cfg_scale = None
    model_name = None
    scheduler = None
    seed = None
    width = None
    height = None
    rand_device = None
    strength = None
    init_image = None
    positive_prompt = None
    negative_prompt = None
    imported_app_version = None

    def to_json(self):
        """Convert the active instance to json format."""
        prop_dict = {}
        prop_dict["generation_mode"] = self.generation_mode
        # dont render prompt nodes if neither are set to avoid the ui thinking it can set them
        # if at least one exists, render them both, but use empty string instead of None if one of them is empty
        # this allows the field that is empty to actually be cleared byt he UI instead of leaving the previous value
        if self.positive_prompt or self.negative_prompt:
            prop_dict["positive_prompt"] = "" if self.positive_prompt is None else self.positive_prompt
            prop_dict["negative_prompt"] = "" if self.negative_prompt is None else self.negative_prompt
        prop_dict["width"] = self.width
        prop_dict["height"] = self.height
        # only render seed if it has a value to avoid ui thinking it can set this and then error
        if self.seed:
            prop_dict["seed"] = self.seed
        prop_dict["rand_device"] = self.rand_device
        prop_dict["cfg_scale"] = self.cfg_scale
        prop_dict["steps"] = self.steps
        prop_dict["scheduler"] = self.scheduler
        prop_dict["clip_skip"] = 0
        prop_dict["model"] = {}
        prop_dict["model"]["model_name"] = self.model_name
        prop_dict["model"]["base_model"] = None
        prop_dict["controlnets"] = []
        prop_dict["loras"] = []
        prop_dict["vae"] = None
        prop_dict["strength"] = self.strength
        prop_dict["init_image"] = self.init_image
        prop_dict["positive_style_prompt"] = None
        prop_dict["negative_style_prompt"] = None
        prop_dict["refiner_model"] = None
        prop_dict["refiner_cfg_scale"] = None
        prop_dict["refiner_steps"] = None
        prop_dict["refiner_scheduler"] = None
        prop_dict["refiner_aesthetic_store"] = None
        prop_dict["refiner_start"] = None
        prop_dict["imported_app_version"] = self.imported_app_version

        return json.dumps(prop_dict)


class InvokeAIMetadataParser:
    """Parses strings with json data  to find Invoke AI core metadata properties."""

    def __init__(self):
        pass

    def parse_meta_tag_dream(self, dream_string):
        """Take as input an png metadata json node for the 'dream' field variant from prior to 1.15"""
        props = InvokeAIMetadata()

        props.imported_app_version = "pre1.15"
        seed_match = re.search("-S\\s*(\\d+)", dream_string)
        if seed_match is not None:
            try:
                props.seed = int(seed_match[1])
            except ValueError:
                props.seed = None
            raw_prompt = re.sub("(-S\\s*\\d+)", "", dream_string)
        else:
            raw_prompt = dream_string

        pos_prompt, neg_prompt = self.split_prompt(raw_prompt)

        props.positive_prompt = pos_prompt
        props.negative_prompt = neg_prompt

        return props

    def parse_meta_tag_sd_metadata(self, tag_value):
        """Take as input an png metadata json node for the 'sd-metadata' field variant from 1.15 through 2.3.5 post 2"""
        props = InvokeAIMetadata()

        props.imported_app_version = tag_value.get("app_version")
        props.model_name = tag_value.get("model_weights")
        img_node = tag_value.get("image")
        if img_node is not None:
            props.generation_mode = img_node.get("type")
            props.width = img_node.get("width")
            props.height = img_node.get("height")
            props.seed = img_node.get("seed")
            props.rand_device = "cuda"  # hardcoded since all generations pre 3.0 used cuda random noise instead of cpu
            props.cfg_scale = img_node.get("cfg_scale")
            props.steps = img_node.get("steps")
            props.scheduler = self.map_scheduler(img_node.get("sampler"))
            props.strength = img_node.get("strength")
            if props.strength is None:
                props.strength = img_node.get("strength_steps")  # try second name for this property
            props.init_image = img_node.get("init_image_path")
            if props.init_image is None:  # try second name for this property
                props.init_image = img_node.get("init_img")
            # remove the path info from init_image so if we move the init image, it will be correctly relative in the new location
            if props.init_image is not None:
                props.init_image = os.path.basename(props.init_image)
            raw_prompt = img_node.get("prompt")
            if isinstance(raw_prompt, list):
                raw_prompt = raw_prompt[0].get("prompt")

            props.positive_prompt, props.negative_prompt = self.split_prompt(raw_prompt)

        return props

    def parse_meta_tag_invokeai(self, tag_value):
        """Take as input an png metadata json node for the 'invokeai' field variant from 3.0.0 beta 1 through 5"""
        props = InvokeAIMetadata()

        props.imported_app_version = "3.0.0 or later"
        props.generation_mode = tag_value.get("type")
        if props.generation_mode is not None:
            props.generation_mode = props.generation_mode.replace("t2l", "txt2img").replace("l2l", "img2img")

        props.width = tag_value.get("width")
        props.height = tag_value.get("height")
        props.seed = tag_value.get("seed")
        props.cfg_scale = tag_value.get("cfg_scale")
        props.steps = tag_value.get("steps")
        props.scheduler = tag_value.get("scheduler")
        props.strength = tag_value.get("strength")
        props.positive_prompt = tag_value.get("positive_conditioning")
        props.negative_prompt = tag_value.get("negative_conditioning")

        return props

    def map_scheduler(self, old_scheduler):
        """Convert the legacy sampler names to matching 3.0 schedulers"""

        # this was more elegant as a case statement, but that's not available in python 3.9
        if old_scheduler is None:
            return None
        scheduler_map = dict(
            ddim="ddim",
            plms="pnmd",
            k_lms="lms",
            k_dpm_2="kdpm_2",
            k_dpm_2_a="kdpm_2_a",
            dpmpp_2="dpmpp_2s",
            k_dpmpp_2="dpmpp_2m",
            k_dpmpp_2_a=None,  # invalid, in 2.3.x, selecting this sample would just fallback to last run or plms if new session
            k_euler="euler",
            k_euler_a="euler_a",
            k_heun="heun",
        )
        return scheduler_map.get(old_scheduler)

    def split_prompt(self, raw_prompt: str):
        """Split the unified prompt strings by extracting all negative prompt blocks out into the negative prompt."""
        if raw_prompt is None:
            return "", ""
        raw_prompt_search = raw_prompt.replace("\r", "").replace("\n", "")
        matches = re.findall(r"\[(.+?)\]", raw_prompt_search)
        if len(matches) > 0:
            negative_prompt = ""
            if len(matches) == 1:
                negative_prompt = matches[0].strip().strip(",")
            else:
                for match in matches:
                    negative_prompt += f"({match.strip().strip(',')})"
            positive_prompt = re.sub(r"(\[.+?\])", "", raw_prompt_search).strip()
        else:
            positive_prompt = raw_prompt_search.strip()
            negative_prompt = ""

        return positive_prompt, negative_prompt


class DatabaseMapper:
    """Class to abstract database functionality."""

    def __init__(self, database_path, database_backup_dir):
        self.database_path = database_path
        self.database_backup_dir = database_backup_dir
        self.connection = None
        self.cursor = None

    def connect(self):
        """Open connection to the database."""
        self.connection = sqlite3.connect(self.database_path)
        self.cursor = self.connection.cursor()

    def get_board_names(self):
        """Get a list of the current board names from the database."""
        sql_get_board_name = "SELECT board_name FROM boards"
        self.cursor.execute(sql_get_board_name)
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def does_image_exist(self, image_name):
        """Check database if a image name already exists and return a boolean."""
        sql_get_image_by_name = f"SELECT image_name FROM images WHERE image_name='{image_name}'"
        self.cursor.execute(sql_get_image_by_name)
        rows = self.cursor.fetchall()
        return True if len(rows) > 0 else False

    def add_new_image_to_database(self, filename, width, height, metadata, modified_date_string):
        """Add an image to the database."""
        sql_add_image = f"""INSERT INTO images (image_name, image_origin, image_category, width, height, session_id, node_id, metadata, is_intermediate, created_at, updated_at)
VALUES ('{filename}', 'internal', 'general', {width}, {height}, null, null, '{metadata}', 0, '{modified_date_string}', '{modified_date_string}')"""
        self.cursor.execute(sql_add_image)
        self.connection.commit()

    def get_board_id_with_create(self, board_name):
        """Get the board id for supplied name, and create the board if one does not exist."""
        sql_find_board = f"SELECT board_id FROM boards WHERE board_name='{board_name}' COLLATE NOCASE"
        self.cursor.execute(sql_find_board)
        rows = self.cursor.fetchall()
        if len(rows) > 0:
            return rows[0][0]
        else:
            board_date_string = datetime.datetime.utcnow().date().isoformat()
            new_board_id = str(uuid.uuid4())
            sql_insert_board = f"INSERT INTO boards (board_id, board_name, created_at, updated_at) VALUES ('{new_board_id}', '{board_name}', '{board_date_string}', '{board_date_string}')"
            self.cursor.execute(sql_insert_board)
            self.connection.commit()
            return new_board_id

    def add_image_to_board(self, filename, board_id):
        """Add an image mapping to a board."""
        add_datetime_str = datetime.datetime.utcnow().isoformat()
        sql_add_image_to_board = f"""INSERT INTO board_images (board_id, image_name, created_at, updated_at)
            VALUES ('{board_id}', '{filename}', '{add_datetime_str}', '{add_datetime_str}')"""
        self.cursor.execute(sql_add_image_to_board)
        self.connection.commit()

    def disconnect(self):
        """Disconnect from the db, cleaning up connections and cursors."""
        if self.cursor is not None:
            self.cursor.close()
        if self.connection is not None:
            self.connection.close()

    def backup(self, timestamp_string):
        """Take a backup of the database."""
        if not os.path.exists(self.database_backup_dir):
            print(f"Database backup directory {self.database_backup_dir} does not exist -> creating...", end="")
            os.makedirs(self.database_backup_dir)
            print("Done!")
        database_backup_path = os.path.join(self.database_backup_dir, f"backup-{timestamp_string}-invokeai.db")
        print(f"Making DB Backup at {database_backup_path}...", end="")
        shutil.copy2(self.database_path, database_backup_path)
        print("Done!")


class MediaImportProcessor:
    """Containing class for script functionality."""

    def __init__(self):
        pass

    board_name_id_map = {}

    def get_import_file_list(self):
        """Ask the user for the import folder and scan for the list of files to return."""
        while True:
            default = ""
            while True:
                import_dir = os.path.expanduser(
                    prompt(
                        "Inputs: Specify absolute path containing InvokeAI .png images to import: ",
                        completer=PathCompleter(expanduser=True, only_directories=True),
                        default=default,
                    )
                )
                if len(import_dir) > 0 and Path(import_dir).is_dir():
                    break
                default = import_dir

            recurse_directories = (
                (prompt("Include files from subfolders recursively [yN]? ").strip() or "N").upper().startswith("N")
            )
            if recurse_directories:
                is_recurse = False
                matching_file_list = glob.glob(import_dir + "/*.png", recursive=False)
            else:
                is_recurse = True
                matching_file_list = glob.glob(import_dir + "/**/*.png", recursive=True)

            if len(matching_file_list) > 0:
                return import_dir, is_recurse, matching_file_list
            else:
                print(f"The specific path {import_dir} exists, but does not contain .png files!")

    def get_file_details(self, filepath):
        """Retrieve the embedded metedata fields and dimensions from an image file."""
        with PIL.Image.open(filepath) as img:
            img.load()
            png_width, png_height = img.size
            img_info = img.info
        return img_info, png_width, png_height

    def select_board_option(self, board_names, timestamp_string):
        """Allow the user to choose how a board is selected for imported files."""
        while True:
            print("\r\nOptions for board selection for imported images:")
            print(f"1) Select an existing board name. (found {len(board_names)})")
            print("2) Specify a board name to create/add to.")
            print("3) Create/add to board named 'IMPORT'.")
            print(
                f"4) Create/add to board named 'IMPORT' with the current datetime string appended (.e.g IMPORT_{timestamp_string})."
            )
            print(
                "5) Create/add to board named 'IMPORT' with a the original file app_version appended (.e.g IMPORT_2.2.5)."
            )
            input_option = input("Specify desired board option: ")
            # This was more elegant as a case statement, but not supported in python 3.9
            if input_option == "1":
                if len(board_names) < 1:
                    print("\r\nThere are no existing board names to choose from. Select another option!")
                    continue
                board_name = self.select_item_from_list(
                    board_names, "board name", True, "Cancel, go back and choose a different board option."
                )
                if board_name is not None:
                    return board_name
            elif input_option == "2":
                while True:
                    board_name = input("Specify new/existing board name: ")
                    if board_name:
                        return board_name
            elif input_option == "3":
                return "IMPORT"
            elif input_option == "4":
                return f"IMPORT_{timestamp_string}"
            elif input_option == "5":
                return "IMPORT_APPVERSION"

    def select_item_from_list(self, items, entity_name, allow_cancel, cancel_string):
        """A general function to render a list of items to select in the console, prompt the user for a selection and ensure a valid entry is selected."""
        print(f"Select a {entity_name.lower()} from the following list:")
        index = 1
        for item in items:
            print(f"{index}) {item}")
            index += 1
        if allow_cancel:
            print(f"{index}) {cancel_string}")
        while True:
            try:
                option_number = int(input("Specify number of selection: "))
            except ValueError:
                continue
            if allow_cancel and option_number == index:
                return None
            if option_number >= 1 and option_number <= len(items):
                return items[option_number - 1]

    def import_image(self, filepath: str, board_name_option: str, db_mapper: DatabaseMapper, config: Config):
        """Import a single file by its path"""
        parser = InvokeAIMetadataParser()
        file_name = os.path.basename(filepath)
        file_destination_path = os.path.join(config.outputs_path, file_name)

        print("===============================================================================")
        print(f"Importing {filepath}")

        # check destination to see if the file was previously imported
        if os.path.exists(file_destination_path):
            print("File already exists in the destination, skipping!")
            ImportStats.count_skipped_file_exists += 1
            return

        # check if file name is already referenced in the database
        if db_mapper.does_image_exist(file_name):
            print("A reference to a file with this name already exists in the database, skipping!")
            ImportStats.count_skipped_db_exists += 1
            return

        # load image info and dimensions
        img_info, png_width, png_height = self.get_file_details(filepath)

        # parse metadata
        destination_needs_meta_update = True
        log_version_note = "(Unknown)"
        if "invokeai_metadata" in img_info:
            # for the latest, we will just re-emit the same json, no need to parse/modify
            converted_field = None
            latest_json_string = img_info.get("invokeai_metadata")
            log_version_note = "3.0.0+"
            destination_needs_meta_update = False
        else:
            if "sd-metadata" in img_info:
                converted_field = parser.parse_meta_tag_sd_metadata(json.loads(img_info.get("sd-metadata")))
            elif "invokeai" in img_info:
                converted_field = parser.parse_meta_tag_invokeai(json.loads(img_info.get("invokeai")))
            elif "dream" in img_info:
                converted_field = parser.parse_meta_tag_dream(img_info.get("dream"))
            elif "Dream" in img_info:
                converted_field = parser.parse_meta_tag_dream(img_info.get("Dream"))
            else:
                converted_field = InvokeAIMetadata()
                destination_needs_meta_update = False
                print("File does not have metadata from known Invoke AI versions, add only, no update!")

            # use the loaded img dimensions if the metadata didnt have them
            if converted_field.width is None:
                converted_field.width = png_width
            if converted_field.height is None:
                converted_field.height = png_height

            log_version_note = converted_field.imported_app_version if converted_field else "NoVersion"
            log_version_note = log_version_note or "NoVersion"

            latest_json_string = converted_field.to_json()

        print(f"From Invoke AI Version {log_version_note} with dimensions {png_width} x {png_height}.")

        # if metadata needs update, then update metdata and copy in one shot
        if destination_needs_meta_update:
            print("Updating metadata while copying...", end="")
            self.update_file_metadata_while_copying(
                filepath, file_destination_path, "invokeai_metadata", latest_json_string
            )
            print("Done!")
        else:
            print("No metadata update necessary, copying only...", end="")
            shutil.copy2(filepath, file_destination_path)
            print("Done!")

        # create thumbnail
        print("Creating thumbnail...", end="")
        thumbnail_path = os.path.join(config.thumbnail_path, os.path.splitext(file_name)[0]) + ".webp"
        thumbnail_size = 256, 256
        with PIL.Image.open(filepath) as source_image:
            source_image.thumbnail(thumbnail_size)
            source_image.save(thumbnail_path, "webp")
        print("Done!")

        # finalize the dynamic board name if there is an APPVERSION token in it.
        if converted_field is not None:
            board_name = board_name_option.replace("APPVERSION", converted_field.imported_app_version or "NoVersion")
        else:
            board_name = board_name_option.replace("APPVERSION", "Latest")

        # maintain a map of alrady created/looked up ids to avoid DB queries
        print("Finding/Creating board...", end="")
        if board_name in self.board_name_id_map:
            board_id = self.board_name_id_map[board_name]
        else:
            board_id = db_mapper.get_board_id_with_create(board_name)
            self.board_name_id_map[board_name] = board_id
        print("Done!")

        # add image to db
        print("Adding image to database......", end="")
        modified_time = datetime.datetime.utcfromtimestamp(os.path.getmtime(filepath))
        db_mapper.add_new_image_to_database(file_name, png_width, png_height, latest_json_string, modified_time)
        print("Done!")

        # add image to board
        print("Adding image to board......", end="")
        db_mapper.add_image_to_board(file_name, board_id)
        print("Done!")

        ImportStats.count_imported += 1
        if log_version_note in ImportStats.count_imported_by_version:
            ImportStats.count_imported_by_version[log_version_note] += 1
        else:
            ImportStats.count_imported_by_version[log_version_note] = 1

    def update_file_metadata_while_copying(self, filepath, file_destination_path, tag_name, tag_value):
        """Perform a metadata update with save to a new destination which accomplishes a copy while updating metadata."""
        with PIL.Image.open(filepath) as target_image:
            existing_img_info = target_image.info
            metadata = PIL.PngImagePlugin.PngInfo()
            # re-add any existing invoke ai tags unless they are the one we are trying to add
            for key in existing_img_info:
                if key != tag_name and key in ("dream", "Dream", "sd-metadata", "invokeai", "invokeai_metadata"):
                    metadata.add_text(key, existing_img_info[key])
            metadata.add_text(tag_name, tag_value)
            target_image.save(file_destination_path, pnginfo=metadata)

    def process(self):
        """Begin main processing."""

        print("===============================================================================")
        print("This script will import images generated by earlier versions of")
        print("InvokeAI into the currently installed root directory:")
        print(f"   {app_config.root_path}")
        print("If this is not what you want to do, type ctrl-C now to cancel.")

        # load config
        print("===============================================================================")
        print("= Configuration & Settings")

        config = Config()
        config.find_and_load()
        db_mapper = DatabaseMapper(config.database_path, config.database_backup_dir)
        db_mapper.connect()

        import_dir, is_recurse, import_file_list = self.get_import_file_list()
        ImportStats.count_source_files = len(import_file_list)

        board_names = db_mapper.get_board_names()
        board_name_option = self.select_board_option(board_names, config.TIMESTAMP_STRING)

        print("\r\n===============================================================================")
        print("= Import Settings Confirmation")

        print()
        print(f"Database File Path               : {config.database_path}")
        print(f"Outputs/Images Directory         : {config.outputs_path}")
        print(f"Import Image Source Directory    : {import_dir}")
        print(f"  Recurse Source SubDirectories  : {'Yes' if is_recurse else 'No'}")
        print(f"Count of .png file(s) found      : {len(import_file_list)}")
        print(f"Board name option specified      : {board_name_option}")
        print(f"Database backup will be taken at : {config.database_backup_dir}")

        print("\r\nNotes about the import process:")
        print("- Source image files will not be modified, only copied to the outputs directory.")
        print("- If the same file name already exists in the destination, the file will be skipped.")
        print("- If the same file name already has a record in the database, the file will be skipped.")
        print("- Invoke AI metadata tags will be updated/written into the imported copy only.")
        print(
            "- On the imported copy, only Invoke AI known tags (latest and legacy) will be retained (dream, sd-metadata, invokeai, invokeai_metadata)"
        )
        print(
            "- A property 'imported_app_version' will be added to metadata that can be viewed in the UI's metadata viewer."
        )
        print(
            "- The new 3.x InvokeAI outputs folder structure is flat so recursively found source imges will all be placed into the single outputs/images folder."
        )

        while True:
            should_continue = prompt("\nDo you wish to continue with the import [Yn] ? ").lower() or "y"
            if should_continue == "n":
                print("\r\nCancelling Import")
                return
            elif should_continue == "y":
                print()
                break

        db_mapper.backup(config.TIMESTAMP_STRING)

        print()
        ImportStats.time_start = datetime.datetime.utcnow()

        for filepath in import_file_list:
            try:
                self.import_image(filepath, board_name_option, db_mapper, config)
            except sqlite3.Error as sql_ex:
                print(f"A database related exception was found processing {filepath}, will continue to next file. ")
                print("Exception detail:")
                print(sql_ex)
                ImportStats.count_file_errors += 1
            except Exception as ex:
                print(f"Exception processing {filepath}, will continue to next file. ")
                print("Exception detail:")
                print(ex)
                ImportStats.count_file_errors += 1

        print("\r\n===============================================================================")
        print(f"= Import Complete - Elpased Time: {ImportStats.get_elapsed_time_string()}")
        print()
        print(f"Source File(s)                          : {ImportStats.count_source_files}")
        print(f"Total Imported                          : {ImportStats.count_imported}")
        print(f"Skipped b/c file already exists on disk : {ImportStats.count_skipped_file_exists}")
        print(f"Skipped b/c file already exists in db   : {ImportStats.count_skipped_db_exists}")
        print(f"Errors during import                    : {ImportStats.count_file_errors}")
        if ImportStats.count_imported > 0:
            print("\r\nBreakdown of imported files by version:")
            for key, version in ImportStats.count_imported_by_version.items():
                print(f"  {key:20} : {version}")


def main():
    try:
        processor = MediaImportProcessor()
        processor.process()
    except KeyboardInterrupt:
        print("\r\n\r\nUser cancelled execution.")


if __name__ == "__main__":
    main()
