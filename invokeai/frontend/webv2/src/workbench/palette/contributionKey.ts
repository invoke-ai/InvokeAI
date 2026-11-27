import type { WidgetContributionSource } from '@workbench/widgetContracts';

/** Includes widget source identity; the tuple is mirrored here to keep the palette's lazy boundary. */
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
