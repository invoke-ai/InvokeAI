/**
 * Registry for deferred widget chunks; the architecture gate checks completeness and measurements derive
 * widget:<id> identifiers.
 */
export const WIDGET_IMPLEMENTATION_PATTERN = /^src\/workbench\/widgets\/([^/]+)\/implementation\.ts$/;

export const WIDGET_SOURCES = new Map([
  ['src/workbench/widgets/autosave-status/implementation.ts', 'autosave-status'],
  ['src/workbench/widgets/canvas/implementation.ts', 'canvas'],
  ['src/workbench/widgets/diagnostics/implementation.ts', 'diagnostics'],
  ['src/workbench/widgets/image-map/implementation.ts', 'image-map'],
  ['src/workbench/widgets/layers/implementation.ts', 'layers'],
  ['src/workbench/widgets/notifications/implementation.ts', 'notifications'],
  ['src/workbench/widgets/preview/implementation.ts', 'preview'],
  ['src/workbench/widgets/project/implementation.ts', 'project'],
  ['src/workbench/widgets/queue-status/implementation.ts', 'queue-status'],
  ['src/workbench/widgets/server-status/implementation.ts', 'server-status'],
  ['src/features/gallery/widget.ts', 'gallery'],
  ['src/features/generation/widget.ts', 'generate'],
  ['src/features/queue/ui/index.ts', 'queue'],
  ['src/features/upscale/widget.ts', 'upscale'],
  ['src/features/workflow/ui/implementation.ts', 'workflow'],
]);

export const getWidgetId = (source) => WIDGET_SOURCES.get(source) ?? null;
