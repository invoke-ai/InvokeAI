# Settings contributions

Paths are relative to the webv2 `src/` directory.

Owners declare settings; Workbench presents them. Platform UI owns domain-neutral field metadata and controls, not setting values or persistence. Widget manifest exposes a `settings` contribution: localized field metadata, ordered `quick` subset, deferred field binding loader. Full dialog includes every declared field. Bindings keep owner's normalization and command path, including Gallery's size/density conversion and project/instance fencing.

Workbench composes widget and application contributions into a metadata-only catalog, shared by search, dialog navigation, quick popovers, and command-palette destinations. Features publish contribution metadata through registered public entries; loading metadata must not load their UI or start data queries. Application-owned additions can join a widget section without changing owner of their values. Deferred resources cache code only; field subscriptions exist only while their controls are mounted.

Section placement does not determine storage lifetime. Account/install preferences, project settings, widget-instance values, and server configuration keep existing persistence and authorization. Each surface lists only settings it can edit: Launchpad Preferences page omits project and widget settings; workbench dialog omits widget settings for widget types the open project lacks. Shell resolves explicit project/instance targets and disposes target-bound editing when projects change. Owner bindings fence writes against stale account/project lifetimes.

Settings host owns dialog's presence lifecycle and keeps its root mounted through closing. Body deferred until first opening, stays present during exit, unmounts afterward. Quick-popover handoff carries both instance target and original gear for focus restoration.

Before section navigation becomes available, deferred dialog prepares binding/editor code without mounting editors or starting their queries. Section and scroll position survive closing within account lifetime. Shell renders a shared modified-setting indicator; bindings decide whether a value differs from owner’s canonical default. Quick settings use shared arrowed popover with inner scroll area; dialog selects keep their bounded scroll areas inside modal focus scope.

Production chunking keeps catalog and manifest metadata separate from editor boot UI. Build gate checks that Launchpad’s palette and settings dialog do not statically include editor application or widget renderer. Shared settings controls stay in common UI chunk; owner field bindings and custom editors stay deferred.
