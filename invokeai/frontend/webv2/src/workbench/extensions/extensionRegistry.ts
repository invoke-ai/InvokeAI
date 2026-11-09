import type {
  WidgetCommandApi,
  WidgetCommandContribution,
  WidgetCommandPaletteApi,
  WidgetCommandPaletteContribution,
  WidgetContributionSource,
  WidgetHotkeyApi,
  WidgetHotkeyContribution,
  WidgetSearchApi,
  WidgetSearchProvider,
} from '@workbench/widgetContracts';

import { createCollectionStore, type CollectionStore } from '@platform/state/externalStore';

import { areWidgetContributionSourcesEqual, getWidgetContributionRegistrationKey } from './contributionSource';

export interface ExtensionContributionStores {
  commands: CollectionStore<WidgetCommandContribution>;
  hotkeys: CollectionStore<WidgetHotkeyContribution>;
  palette: CollectionStore<WidgetCommandPaletteContribution>;
  search: CollectionStore<WidgetSearchProvider>;
}

export interface ExtensionRegistry {
  commands: WidgetCommandApi;
  hotkeys: WidgetHotkeyApi;
  palette: WidgetCommandPaletteApi;
  search: WidgetSearchApi;
  /** Read side for shell surfaces (hotkey runtime and palette/search UI). */
  stores: ExtensionContributionStores;
}

/** Construct one registry per Workbench mount so contribution state shares its lifetime. */
export const createExtensionRegistry = (): ExtensionRegistry => {
  const stores: ExtensionContributionStores = {
    commands: createCollectionStore<WidgetCommandContribution>(),
    hotkeys: createCollectionStore<WidgetHotkeyContribution>(),
    palette: createCollectionStore<WidgetCommandPaletteContribution>(),
    search: createCollectionStore<WidgetSearchProvider>(),
  };

  const findCommand = (
    commandId: string,
    source: WidgetContributionSource | null = null
  ): WidgetCommandContribution | undefined =>
    source
      ? (stores.commands.findLatest(
          (candidate) => candidate.id === commandId && areWidgetContributionSourcesEqual(candidate.source, source)
        ) ?? stores.commands.findLatest((candidate) => candidate.id === commandId && !candidate.source))
      : stores.commands.findLatest((candidate) => candidate.id === commandId && !candidate.source);

  return {
    commands: {
      execute(commandId, ...args) {
        const command = findCommand(commandId);

        return Promise.resolve(command ? command.handler(...args) : undefined);
      },
      executeForSource(commandId, source, ...args) {
        const command = findCommand(commandId, source);

        return Promise.resolve(command ? command.handler(...args) : undefined);
      },
      register(command) {
        return stores.commands.register(command, getWidgetContributionRegistrationKey(command.id, command.source));
      },
    },
    hotkeys: {
      register(hotkey) {
        return stores.hotkeys.register(hotkey, getWidgetContributionRegistrationKey(hotkey.id, hotkey.source));
      },
    },
    palette: {
      register(entry) {
        return stores.palette.register(entry, getWidgetContributionRegistrationKey(entry.commandId, entry.source));
      },
    },
    search: {
      registerProvider(provider) {
        // Each registration gets a fresh identity. Consumers can distinguish a
        // replacement even when an extension reuses its id/source and omits a
        // contextKey.
        const registration = { ...provider };

        return stores.search.register(
          registration,
          getWidgetContributionRegistrationKey(registration.id, registration.source)
        );
      },
    },
    stores,
  };
};
