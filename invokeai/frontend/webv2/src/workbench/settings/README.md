# Settings contributions

Paths are relative to the webv2 `src/` directory.

Settings are declared by their owners and presented by Workbench. Platform UI owns domain-neutral field metadata and controls; it does not own setting values or persistence. A widget manifest exposes a `settings` contribution containing localized field metadata, an ordered `quick` subset, and a deferred field binding loader. The full dialog includes every declared field. Bindings retain the owner's normalization and command path, including Gallery's size/density conversion and project/instance fencing.

Workbench composes widget and application contributions into a metadata-only catalog. Search, dialog navigation, quick popovers, and command-palette destinations share that catalog. Features publish contribution metadata through registered public entries; loading metadata must not load their UI or start data queries. Application-owned additions can join a widget section without changing the owner of their values. Deferred resources cache code only; field subscriptions exist only while their controls are mounted.

Section placement does not determine storage lifetime. Account/install preferences, project settings, widget-instance values, and server configuration retain their existing persistence and authorization. Each surface lists only settings it can edit: the Launchpad Preferences page leaves out project and widget settings, and the workbench dialog leaves out widget settings for widget types the open project lacks. The shell resolves explicit project/instance targets and disposes target-bound editing when projects change. Owner bindings fence writes against stale account/project lifetimes.

The settings host owns the dialog's presence lifecycle and keeps its root mounted through closing. The body is deferred until first opening, remains present during exit, and unmounts afterward. Quick-popover handoff carries both the instance target and the original gear for focus restoration.

Before section navigation becomes available, the deferred dialog prepares binding/editor code without mounting editors or starting their queries. Section and scroll position survive closing within the account lifetime. The shell renders a shared modified-setting indicator; bindings determine whether a value differs from the owner’s canonical default. Quick settings use the shared arrowed popover with an inner scroll area, and dialog selects keep their bounded scroll areas inside the modal focus scope.

Production chunking keeps the catalog and manifest metadata separate from editor boot UI. The build gate checks that Launchpad’s palette and settings dialog do not statically include the editor application or widget renderer. Shared settings controls remain in the common UI chunk; owner field bindings and custom editors remain deferred.
