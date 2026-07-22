import type {
  WidgetCommandPaletteContribution,
  WidgetContributionSource,
  WidgetSearchProvider,
} from '@workbench/widgetContracts';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';

type ExecuteExtensionCommand = (
  commandId: string,
  source: WidgetContributionSource | null,
  resultId?: string
) => unknown;

export const buildExtensionPaletteEntry = (
  contribution: WidgetCommandPaletteContribution,
  execute: ExecuteExtensionCommand
): PaletteEntry => ({
  group: 'Commands',
  id: getPaletteContributionKey('command', contribution.commandId, contribution.source),
  isPersistentRecent: true,
  keywords: contribution.keywords?.join(' '),
  run: () => execute(contribution.commandId, contribution.source ?? null),
  title: contribution.title,
});

export const createExtensionSearchProvider = (
  provider: WidgetSearchProvider,
  execute: ExecuteExtensionCommand
): PaletteSearchProvider => {
  const providerKey = getPaletteContributionKey('provider', provider.id, provider.source);

  return {
    contextKey: providerKey,
    label: provider.label,
    providerKey,
    search: async (query) => {
      const results = await provider.search(query);

      return results.map<PaletteEntry>((result) => ({
        group: provider.label,
        id: getPaletteContributionKey('provider-result', `${providerKey}:${result.id}`, provider.source),
        isPersistentRecent: false,
        run: () => (result.commandId ? execute(result.commandId, provider.source ?? null, result.id) : undefined),
        subtitle: result.subtitle,
        title: result.title,
      }));
    },
  };
};
