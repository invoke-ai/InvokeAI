# Copyright 2011 Yesudeep Mangalapilly <yesudeep@gmail.com>
# Copyright 2012 Google, Inc & contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
:module: watchdog.events
:synopsis: File system events and event handlers.
:author: yesudeep@google.com (Yesudeep Mangalapilly)
:author: contact@tiger-222.fr (Mickaël Schoentgen)

Event Classes
-------------
.. autoclass:: FileSystemEvent
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: FileSystemMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirMovedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileModifiedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirModifiedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileCreatedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileClosedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileOpenedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirCreatedEvent
   :members:
   :show-inheritance:

.. autoclass:: FileDeletedEvent
   :members:
   :show-inheritance:

.. autoclass:: DirDeletedEvent
   :members:
   :show-inheritance:


Event Handler Classes
---------------------
.. autoclass:: FileSystemEventHandler
   :members:
   :show-inheritance:

.. autoclass:: PatternMatchingEventHandler
   :members:
   :show-inheritance:

.. autoclass:: RegexMatchingEventHandler
   :members:
   :show-inheritance:

.. autoclass:: LoggingEventHandler
   :members:
   :show-inheritance:

"""

from __future__ import annotations

import logging
import os.path
import re
from dataclasses import dataclass, field
from typing import Optional

from watchdog.utils.patterns import match_any_paths

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"
EVENT_TYPE_CLOSED = "closed"
EVENT_TYPE_OPENED = "opened"


@dataclass(unsafe_hash=True)
class FileSystemEvent:
    """
    Immutable type that represents a file system event that is triggered
    when a change occurs on the monitored file system.

    All FileSystemEvent objects are required to be immutable and hence
    can be used as keys in dictionaries or be added to sets.
    """

    src_path: str
    dest_path: str = ""
    event_type: str = field(default="", init=False)
    is_directory: bool = field(default=False, init=False)

    """
    True if event was synthesized; False otherwise.
    These are events that weren't actually broadcast by the OS, but
    are presumed to have happened based on other, actual events.
    """
    is_synthetic: bool = field(default=False)


class FileSystemMovedEvent(FileSystemEvent):
    """File system event representing any kind of file system movement."""

    event_type = EVENT_TYPE_MOVED


# File events.


class FileDeletedEvent(FileSystemEvent):
    """File system event representing file deletion on the file system."""

    event_type = EVENT_TYPE_DELETED


class FileModifiedEvent(FileSystemEvent):
    """File system event representing file modification on the file system."""

    event_type = EVENT_TYPE_MODIFIED


class FileCreatedEvent(FileSystemEvent):
    """File system event representing file creation on the file system."""

    event_type = EVENT_TYPE_CREATED


class FileMovedEvent(FileSystemMovedEvent):
    """File system event representing file movement on the file system."""


class FileClosedEvent(FileSystemEvent):
    """File system event representing file close on the file system."""

    event_type = EVENT_TYPE_CLOSED


class FileOpenedEvent(FileSystemEvent):
    """File system event representing file close on the file system."""

    event_type = EVENT_TYPE_OPENED


# Directory events.


class DirDeletedEvent(FileSystemEvent):
    """File system event representing directory deletion on the file system."""

    event_type = EVENT_TYPE_DELETED
    is_directory = True


class DirModifiedEvent(FileSystemEvent):
    """
    File system event representing directory modification on the file system.
    """

    event_type = EVENT_TYPE_MODIFIED
    is_directory = True


class DirCreatedEvent(FileSystemEvent):
    """File system event representing directory creation on the file system."""

    event_type = EVENT_TYPE_CREATED
    is_directory = True


class DirMovedEvent(FileSystemMovedEvent):
    """File system event representing directory movement on the file system."""

    is_directory = True


class FileSystemEventHandler:
    """
    Base file system event handler that you can override methods from.
    """

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        self.on_any_event(event)
        {
            EVENT_TYPE_CREATED: self.on_created,
            EVENT_TYPE_DELETED: self.on_deleted,
            EVENT_TYPE_MODIFIED: self.on_modified,
            EVENT_TYPE_MOVED: self.on_moved,
            EVENT_TYPE_CLOSED: self.on_closed,
            EVENT_TYPE_OPENED: self.on_opened,
        }[event.event_type](event)

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """

    def on_moved(self, event: FileSystemEvent) -> None:
        """Called when a file or a directory is moved or renamed.

        :param event:
            Event representing file/directory movement.
        :type event:
            :class:`DirMovedEvent` or :class:`FileMovedEvent`
        """

    def on_created(self, event: FileSystemEvent) -> None:
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Called when a file or directory is deleted.

        :param event:
            Event representing file/directory deletion.
        :type event:
            :class:`DirDeletedEvent` or :class:`FileDeletedEvent`
        """

    def on_modified(self, event: FileSystemEvent) -> None:
        """Called when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """

    def on_closed(self, event: FileSystemEvent) -> None:
        """Called when a file opened for writing is closed.

        :param event:
            Event representing file closing.
        :type event:
            :class:`FileClosedEvent`
        """

    def on_opened(self, event: FileSystemEvent) -> None:
        """Called when a file is opened.

        :param event:
            Event representing file opening.
        :type event:
            :class:`FileOpenedEvent`
        """


class PatternMatchingEventHandler(FileSystemEventHandler):
    """
    Matches given patterns with file paths associated with occurring events.
    """

    def __init__(
        self,
        patterns=None,
        ignore_patterns=None,
        ignore_directories=False,
        case_sensitive=False,
    ):
        super().__init__()

        self._patterns = patterns
        self._ignore_patterns = ignore_patterns
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def patterns(self):
        """
        (Read-only)
        Patterns to allow matching event paths.
        """
        return self._patterns

    @property
    def ignore_patterns(self):
        """
        (Read-only)
        Patterns to ignore matching event paths.
        """
        return self._ignore_patterns

    @property
    def ignore_directories(self):
        """
        (Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
        return self._ignore_directories

    @property
    def case_sensitive(self):
        """
        (Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
        return self._case_sensitive

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        if self.ignore_directories and event.is_directory:
            return

        paths = []
        if hasattr(event, "dest_path"):
            paths.append(os.fsdecode(event.dest_path))
        if event.src_path:
            paths.append(os.fsdecode(event.src_path))

        if match_any_paths(
            paths,
            included_patterns=self.patterns,
            excluded_patterns=self.ignore_patterns,
            case_sensitive=self.case_sensitive,
        ):
            super().dispatch(event)


class RegexMatchingEventHandler(FileSystemEventHandler):
    """
    Matches given regexes with file paths associated with occurring events.
    """

    def __init__(
        self,
        regexes=None,
        ignore_regexes=None,
        ignore_directories=False,
        case_sensitive=False,
    ):
        super().__init__()

        if regexes is None:
            regexes = [r".*"]
        elif isinstance(regexes, str):
            regexes = [regexes]
        if ignore_regexes is None:
            ignore_regexes = []
        if case_sensitive:
            self._regexes = [re.compile(r) for r in regexes]
            self._ignore_regexes = [re.compile(r) for r in ignore_regexes]
        else:
            self._regexes = [re.compile(r, re.I) for r in regexes]
            self._ignore_regexes = [re.compile(r, re.I) for r in ignore_regexes]
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def regexes(self):
        """
        (Read-only)
        Regexes to allow matching event paths.
        """
        return self._regexes

    @property
    def ignore_regexes(self):
        """
        (Read-only)
        Regexes to ignore matching event paths.
        """
        return self._ignore_regexes

    @property
    def ignore_directories(self):
        """
        (Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
        return self._ignore_directories

    @property
    def case_sensitive(self):
        """
        (Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
        return self._case_sensitive

    def dispatch(self, event: FileSystemEvent) -> None:
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        if self.ignore_directories and event.is_directory:
            return

        paths = []
        if hasattr(event, "dest_path"):
            paths.append(os.fsdecode(event.dest_path))
        if event.src_path:
            paths.append(os.fsdecode(event.src_path))

        if any(r.match(p) for r in self.ignore_regexes for p in paths):
            return

        if any(r.match(p) for r in self.regexes for p in paths):
            super().dispatch(event)


class LoggingEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.logger = logger or logging.root

    def on_moved(self, event: FileSystemEvent) -> None:
        super().on_moved(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Moved %s: from %s to %s", what, event.src_path, event.dest_path)

    def on_created(self, event: FileSystemEvent) -> None:
        super().on_created(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Created %s: %s", what, event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        super().on_deleted(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        super().on_modified(event)

        what = "directory" if event.is_directory else "file"
        self.logger.info("Modified %s: %s", what, event.src_path)

    def on_closed(self, event: FileSystemEvent) -> None:
        super().on_closed(event)

        self.logger.info("Closed file: %s", event.src_path)

    def on_opened(self, event: FileSystemEvent) -> None:
        super().on_opened(event)

        self.logger.info("Opened file: %s", event.src_path)


def generate_sub_moved_events(src_dir_path, dest_dir_path):
    """Generates an event list of :class:`DirMovedEvent` and
    :class:`FileMovedEvent` objects for all the files and directories within
    the given moved directory that were moved along with the directory.

    :param src_dir_path:
        The source path of the moved directory.
    :param dest_dir_path:
        The destination path of the moved directory.
    :returns:
        An iterable of file system events of type :class:`DirMovedEvent` and
        :class:`FileMovedEvent`.
    """
    for root, directories, filenames in os.walk(dest_dir_path):
        for directory in directories:
            full_path = os.path.join(root, directory)
            renamed_path = full_path.replace(dest_dir_path, src_dir_path) if src_dir_path else ""
            yield DirMovedEvent(renamed_path, full_path, is_synthetic=True)
        for filename in filenames:
            full_path = os.path.join(root, filename)
            renamed_path = full_path.replace(dest_dir_path, src_dir_path) if src_dir_path else ""
            yield FileMovedEvent(renamed_path, full_path, is_synthetic=True)


def generate_sub_created_events(src_dir_path):
    """Generates an event list of :class:`DirCreatedEvent` and
    :class:`FileCreatedEvent` objects for all the files and directories within
    the given moved directory that were moved along with the directory.

    :param src_dir_path:
        The source path of the created directory.
    :returns:
        An iterable of file system events of type :class:`DirCreatedEvent` and
        :class:`FileCreatedEvent`.
    """
    for root, directories, filenames in os.walk(src_dir_path):
        for directory in directories:
            yield DirCreatedEvent(os.path.join(root, directory), is_synthetic=True)
        for filename in filenames:
            yield FileCreatedEvent(os.path.join(root, filename), is_synthetic=True)
