import type { WidgetContributionSource } from '@workbench/widgetContracts';

/**
 * Include widget source identity so instances can reuse local ids. Mirror the source tuple locally to preserve the
 * palette's lazy boundary.
 */
export const getPaletteContributionKey = (
  kind: 'command' | 'provider' | 'provider-error' | 'provider-result' | 'provider-row' | 'scope' | 'scope-command',
  id: string,
  source?: WidgetContributionSource | null
): string =>
  JSON.stringify([
    'palette',
    kind,
    id,
    source ? ['widget', source.projectId, source.region, source.typeId, source.instanceId] : 'global',
  ]);
