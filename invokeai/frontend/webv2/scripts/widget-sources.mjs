/**
 * Single source of truth for the deferred first-party widget implementation
 * chunks. `check-architecture-performance.mjs` asserts every entry splits into
 * its own chunk and that no manifest source matching WIDGET_IMPLEMENTATION_PATTERN
 * is missing from this registry; `measure-architecture-performance.mjs` derives
 * stable `widget:<id>` script ids from it.
 */
export const WIDGET_IMPLEMENTATION_PATTERN = /^src\/workbench\/widgets\/([^/]+)\/implementation\.ts$/;

export const WIDGET_SOURCES = new Map([
  ['src/workbench/widgets/autosave-status/implementation.ts', 'autosave-status'],
  ['src/workbench/widgets/canvas/implementation.ts', 'canvas'],
  ['src/workbench/widgets/diagnostics/implementation.ts', 'diagnostics'],
  ['src/workbench/widgets/layers/implementation.ts', 'layers'],
  ['src/workbench/widgets/notifications/implementation.ts', 'notifications'],
  ['src/workbench/widgets/preview/implementation.ts', 'preview'],
  ['src/workbench/widgets/project/implementation.ts', 'project'],
  ['src/workbench/widgets/server-status/implementation.ts', 'server-status'],
  ['src/workbench/widgets/version-status/implementation.ts', 'version-status'],
  ['src/features/gallery/widget.ts', 'gallery'],
  ['src/features/generation/widget.ts', 'generate'],
  ['src/features/queue/widget.ts', 'queue'],
  ['src/features/upscale/widget.ts', 'upscale'],
  ['src/features/workflow/ui/implementation.ts', 'workflow'],
]);

export const getWidgetId = (source) => WIDGET_SOURCES.get(source) ?? null;
