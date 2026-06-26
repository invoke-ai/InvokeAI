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
} from '@workbench/types';

import { createCollectionStore } from '@workbench/externalStore';

import { areWidgetContributionSourcesEqual, getWidgetContributionRegistrationKey } from './contributionSource';

const noopDispose = (): void => {};

const commandStore = createCollectionStore<WidgetCommandContribution>();
const hotkeyStore = createCollectionStore<WidgetHotkeyContribution>();
const menuStore = createCollectionStore<WidgetMenuContribution>();
const paletteStore = createCollectionStore<WidgetCommandPaletteContribution>();
const searchStore = createCollectionStore<WidgetSearchProvider>();
const toolbarStore = createCollectionStore<WidgetToolbarContribution>();

const findCommand = (
  commandId: string,
  source: WidgetContributionSource | null = null
): WidgetCommandContribution | undefined =>
  source
    ? (commandStore.findLatest(
        (candidate) => candidate.id === commandId && areWidgetContributionSourcesEqual(candidate.source, source)
      ) ?? commandStore.findLatest((candidate) => candidate.id === commandId && !candidate.source))
    : commandStore.findLatest((candidate) => candidate.id === commandId && !candidate.source);

export const commandApi: WidgetCommandApi = {
  execute(commandId, ...args) {
    const command = findCommand(commandId);

    return Promise.resolve(command ? command.handler(...args) : undefined);
  },
  executeForSource(commandId, source, ...args) {
    const command = findCommand(commandId, source);

    return Promise.resolve(command ? command.handler(...args) : undefined);
  },
  register(command) {
    return commandStore.register(command, getWidgetContributionRegistrationKey(command.id, command.source));
  },
};

export const hotkeyApi: WidgetHotkeyApi = {
  register(hotkey) {
    return hotkeyStore.register(hotkey, getWidgetContributionRegistrationKey(hotkey.id, hotkey.source));
  },
};

export const menuApi: WidgetMenuApi = {
  register(menu) {
    return menuStore.register(menu, getWidgetContributionRegistrationKey(menu.id, menu.source));
  },
};

export const commandPaletteApi: WidgetCommandPaletteApi = {
  register(entry) {
    return paletteStore.register(entry, entry.commandId);
  },
};

export const searchApi: WidgetSearchApi = {
  registerProvider(provider) {
    return searchStore.register(provider, provider.id);
  },
};

export const toolbarApi: WidgetToolbarApi = {
  register(toolbar) {
    return toolbarStore.register(toolbar, toolbar.id);
  },
};

export const extensionContributionStores = {
  commands: commandStore,
  hotkeys: hotkeyStore,
  menus: menuStore,
  palette: paletteStore,
  search: searchStore,
  toolbars: toolbarStore,
};

export const disposeExtensionContribution = noopDispose;
