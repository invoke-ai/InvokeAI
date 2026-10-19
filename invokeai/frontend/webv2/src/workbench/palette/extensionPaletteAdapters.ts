import type {
  WidgetCommandPaletteContribution,
  WidgetContributionSource,
  WidgetSearchProvider,
} from '@workbench/widgetContracts';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import { getObjectIdentity } from './objectIdentity';

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
    contextKey: `${providerKey}:${getObjectIdentity(provider, 'registration')}:${provider.contextKey ?? 'default'}`,
    label: provider.label,
    providerKey,
    // Extensions keep their plain-string search contract; they receive the
    // query text with date tokens stripped and never see the range.
    search: async (query, { signal }) => {
      const results = await provider.search(query.text, { signal });

      return results
        .filter((result) => result.commandId !== undefined)
        .map<PaletteEntry>((result) => ({
          group: provider.label,
          id: getPaletteContributionKey('provider-result', `${providerKey}:${result.id}`, provider.source),
          isPersistentRecent: false,
          run: () => execute(result.commandId!, provider.source ?? null, result.id),
          subtitle: result.subtitle,
          title: result.title,
        }));
    },
  };
};
