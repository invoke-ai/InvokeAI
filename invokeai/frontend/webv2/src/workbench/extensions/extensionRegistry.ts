import type {
  WidgetCommandApi,
  WidgetCommandContribution,
  WidgetCommandPaletteApi,
  WidgetCommandPaletteContribution,
  WidgetContributionSource,
  WidgetHotkeyApi,
  WidgetHotkeyContribution,
  WidgetMenuApi,
  WidgetMenuContribution,
  WidgetSearchApi,
  WidgetSearchProvider,
  WidgetToolbarApi,
  WidgetToolbarContribution,
} from '@workbench/widgetContracts';

import { createCollectionStore, type CollectionStore } from '@platform/state/externalStore';

import { areWidgetContributionSourcesEqual, getWidgetContributionRegistrationKey } from './contributionSource';

export interface ExtensionContributionStores {
  commands: CollectionStore<WidgetCommandContribution>;
  hotkeys: CollectionStore<WidgetHotkeyContribution>;
  menus: CollectionStore<WidgetMenuContribution>;
  palette: CollectionStore<WidgetCommandPaletteContribution>;
  search: CollectionStore<WidgetSearchProvider>;
  toolbars: CollectionStore<WidgetToolbarContribution>;
}

export interface ExtensionRegistry {
  commands: WidgetCommandApi;
  hotkeys: WidgetHotkeyApi;
  menus: WidgetMenuApi;
  palette: WidgetCommandPaletteApi;
  search: WidgetSearchApi;
  /** Read side for shell surfaces (hotkey runtime, palette/search/toolbar UI). */
  stores: ExtensionContributionStores;
  toolbars: WidgetToolbarApi;
}

/**
 * One extension registry is constructed per Workbench mount (see
 * WorkbenchProvider) so contribution state lives and dies with the mount —
 * never in process-wide singletons.
 */
export const createExtensionRegistry = (): ExtensionRegistry => {
  const stores: ExtensionContributionStores = {
    commands: createCollectionStore<WidgetCommandContribution>(),
    hotkeys: createCollectionStore<WidgetHotkeyContribution>(),
    menus: createCollectionStore<WidgetMenuContribution>(),
    palette: createCollectionStore<WidgetCommandPaletteContribution>(),
    search: createCollectionStore<WidgetSearchProvider>(),
    toolbars: createCollectionStore<WidgetToolbarContribution>(),
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
    menus: {
      register(menu) {
        return stores.menus.register(menu, getWidgetContributionRegistrationKey(menu.id, menu.source));
      },
    },
    palette: {
      register(entry) {
        return stores.palette.register(entry, getWidgetContributionRegistrationKey(entry.commandId, entry.source));
      },
    },
    search: {
      registerProvider(provider) {
        return stores.search.register(provider, getWidgetContributionRegistrationKey(provider.id, provider.source));
      },
    },
    stores,
    toolbars: {
      register(toolbar) {
        return stores.toolbars.register(toolbar, getWidgetContributionRegistrationKey(toolbar.id, toolbar.source));
      },
    },
  };
};
