import type { WidgetContributionSource } from '@workbench/widgetContracts';

/**
 * Stable identity for every palette contribution. Widget source identity is
 * part of the key so two instances may safely contribute the same local id.
 * The source tuple intentionally mirrors getWidgetContributionSourceKey;
 * keeping the palette helper self-contained prevents its lazy chunk from
 * pulling extension registry internals into either route's initial graph.
 */
export const getPaletteContributionKey = (
  kind: 'command' | 'provider' | 'provider-result' | 'provider-row' | 'scope',
  id: string,
  source?: WidgetContributionSource | null
): string =>
  JSON.stringify([
    'palette',
    kind,
    id,
    source ? ['widget', source.projectId, source.region, source.typeId, source.instanceId] : 'global',
  ]);
