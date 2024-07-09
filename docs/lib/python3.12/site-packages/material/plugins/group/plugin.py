# Copyright (c) 2016-2024 Martin Donath <martin.donath@squidfunk.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

from __future__ import annotations

import logging

from collections.abc import Callable
from mkdocs.config.config_options import Plugins
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.plugins import BasePlugin, event_priority

from .config import GroupConfig

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Group plugin
class GroupPlugin(BasePlugin[GroupConfig]):
    supports_multiple_instances = True

    # Initialize plugin
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize object attributes
        self.is_serve = False
        self.is_dirty = False

    # Determine whether we're serving the site
    def on_startup(self, *, command, dirty):
        self.is_serve = command == "serve"
        self.is_dirty = dirty

    # If the group is enabled, conditionally load plugins - at first, this might
    # sound easier than it actually is, as we need to jump through some hoops to
    # ensure correct ordering among plugins. We're effectively initializing the
    # plugins that are part of the group after all MkDocs finished initializing
    # all other plugins, so we need to patch the order of the methods. Moreover,
    # we must use MkDocs existing plugin collection, or we might have collisions
    # with other plugins that are not part of the group. As so often, this is a
    # little hacky, but has huge potential making plugin configuration easier.
    # There's one little caveat: the `__init__` and `on_startup` methods of the
    # plugins that are part of the group are called after all other plugins, so
    # the `event_priority` decorator for `on_startup` methods is effectively
    # useless. However, the `on_startup` method is only intended to set up the
    # plugin and doesn't receive anything else than the invoked command and
    # whether we're running a dirty build, so there should be no problems.
    @event_priority(150)
    def on_config(self, config):
        if not self.config.enabled:
            return

        # Retrieve plugin collection from configuration
        option: Plugins = dict(config._schema)["plugins"]
        assert isinstance(option, Plugins)

        # Load all plugins in group
        self.plugins: dict[str, BasePlugin] = {}
        try:
            for name, plugin in self._load(option):
                self.plugins[name] = plugin

        # The plugin could not be loaded, likely because it's not installed or
        # misconfigured, so we raise a plugin error for a nicer error message
        except Exception as e:
            raise PluginError(str(e))

        # Patch order of plugin methods
        for events in option.plugins.events.values():
            self._patch(events, config)

        # Invoke `on_startup` event for plugins in group
        command = "serve" if self.is_serve else "build"
        for method in option.plugins.events["startup"]:
            plugin = self._get_plugin(method)

            # Ensure that we have a method bound to a plugin (and not a hook)
            if plugin and plugin in self.plugins.values():
                method(command = command, dirty = self.is_dirty)

    # -------------------------------------------------------------------------

    # Retrieve plugin instance for bound method or nothing
    def _get_plugin(self, method: Callable):
        return getattr(method, "__self__", None)

    # Retrieve priority of plugin method
    def _get_priority(self, method: Callable):
        return getattr(method, "mkdocs_priority", 0)

    # Retrieve position of plugin
    def _get_position(self, plugin: BasePlugin, config: MkDocsConfig) -> int:
        for at, (_, candidate) in enumerate(config.plugins.items()):
            if plugin == candidate:
                return at

    # -------------------------------------------------------------------------

    # Load plugins that are part of the group
    def _load(self, option: Plugins):
        for name, data in option._parse_configs(self.config.plugins):
            yield option.load_plugin_with_namespace(name, data)

    # -------------------------------------------------------------------------

    # Patch order of plugin methods - all other plugin methods are already in
    # the right order, so we only need to check those that are part of the group
    # and bubble them up into the right location. Some plugin methods may define
    # priorities, so we need to make sure to order correctly within those.
    def _patch(self, methods: list[Callable], config: MkDocsConfig):
        position = self._get_position(self, config)
        for at in reversed(range(1, len(methods))):
            tail = methods[at - 1]
            head = methods[at]

            # Skip if the plugin is not part of the group
            plugin = self._get_plugin(head)
            if not plugin or plugin not in self.plugins.values():
                continue

            # Skip if the previous method has a higher priority than the current
            # one, because we know we can't swap them anyway
            if self._get_priority(tail) > self._get_priority(head):
                continue

            # Ensure that we have a method bound to a plugin (and not a hook)
            plugin = self._get_plugin(tail)
            if not plugin:
                continue

            # Both methods have the same priority, so we check if the ordering
            # of both methods is violated, and if it is, swap them
            if (position < self._get_position(plugin, config)):
                methods[at], methods[at - 1] = tail, head

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

# Set up logging
log = logging.getLogger("mkdocs.material.group")
