/**
 * Identifier for a workbench color theme. The matching palette lives in
 * `theme/themes.ts`; the active id is applied to `<html data-theme>` so the
 * semantic-token conditions in `theme/system.ts` resolve to the right colors.
 */
export type DeveloperLogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export type DeveloperLogNamespace =
  | 'canvas'
  | 'canvas-workflow-integration'
  | 'config'
  | 'dnd'
  | 'events'
  | 'gallery'
  | 'generation'
  | 'metadata'
  | 'models'
  | 'system'
  | 'queue'
  | 'workflows';
