import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Open state for the project switcher menu, so `⌘P` can reach it without the
 * hotkey runtime holding a ref to the trigger.
 */
const INITIAL_SNAPSHOT = { isOpen: false };

export const projectSwitcherStore = createExternalStore<{ isOpen: boolean }>(INITIAL_SNAPSHOT);

registerAccountOwnedResource({
  clear: () => {
    projectSwitcherStore.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'project-switcher',
});

export const openProjectSwitcher = (): void => projectSwitcherStore.setSnapshot({ isOpen: true });

export const setProjectSwitcherOpen = (isOpen: boolean): void => projectSwitcherStore.setSnapshot({ isOpen });
