import type {
  WidgetCommandApi,
  WidgetCommandContribution,
  WidgetCommandPaletteApi,
  WidgetCommandPaletteContribution,
  WidgetHotkeyApi,
  WidgetHotkeyContribution,
  WidgetMenuApi,
  WidgetMenuContribution,
  WidgetSearchApi,
  WidgetSearchProvider,
  WidgetToolbarApi,
  WidgetToolbarContribution,
} from '@workbench/types';

const noopDispose = (): void => {};

class ContributionStore<Contribution> {
  private readonly contributions = new Map<string, Contribution>();
  private readonly listeners = new Set<() => void>();

  register(contribution: Contribution, fallbackId: string): () => void {
    const id = fallbackId;

    this.contributions.set(id, contribution);
    this.notify();

    return () => {
      this.contributions.delete(id);
      this.notify();
    };
  }

  list(): Contribution[] {
    return [...this.contributions.values()];
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);

    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}

const commandStore = new ContributionStore<WidgetCommandContribution>();
const hotkeyStore = new ContributionStore<WidgetHotkeyContribution>();
const menuStore = new ContributionStore<WidgetMenuContribution>();
const paletteStore = new ContributionStore<WidgetCommandPaletteContribution>();
const searchStore = new ContributionStore<WidgetSearchProvider>();
const toolbarStore = new ContributionStore<WidgetToolbarContribution>();

export const commandApi: WidgetCommandApi = {
  execute(commandId, ...args) {
    const command = commandStore.list().find((candidate) => candidate.id === commandId);

    return Promise.resolve(command ? command.handler(...args) : undefined);
  },
  register(command) {
    return commandStore.register(command, command.id);
  },
};

export const hotkeyApi: WidgetHotkeyApi = {
  register(hotkey) {
    return hotkeyStore.register(hotkey, hotkey.id);
  },
};

export const menuApi: WidgetMenuApi = {
  register(menu) {
    return menuStore.register(menu, menu.id);
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
