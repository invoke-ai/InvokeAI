import { logger } from 'app/logging/logger';
import { createDeferredPromise, type Deferred } from 'common/util/createDeferredPromise';
import { parseify } from 'common/util/serialize';
import type { DockviewApi, GridviewApi, IDockviewPanel, IGridviewPanel } from 'dockview';
import { GridviewPanel } from 'dockview';
import { debounce } from 'es-toolkit';
import type { Serializable, TabName } from 'features/ui/store/uiTypes';
import type { Atom } from 'nanostores';
import { atom } from 'nanostores';

import {
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SWITCH_TABS_FAKE_DELAY_MS,
} from './shared';

const log = logger('system');

type PanelType = IGridviewPanel | IDockviewPanel;

/**
 * An object that represents a promise that is waiting for a panel to be registered and ready.
 *
 * It includes a deferred promise that can be resolved or rejected, and a timeout ID.
 */
type Waiter = {
  deferred: Deferred<void>;
  timeoutId: ReturnType<typeof setTimeout> | null;
};

/**
 * The API exposed by the application to manage navigation and panel states.
 */
export type NavigationAppApi = {
  /**
   * API to manage the currently active tab in the application.
   */
  activeTab: {
    get: () => TabName;
    set: (tab: TabName) => void;
  };
  /**
   * API to manage the storage of panel states.
   */
  storage: {
    get: (id: string) => Serializable | undefined;
    set: (id: string, state: Serializable) => void;
    delete: (id: string) => void;
  };
};

export class NavigationApi {
  /**
   * Map of registered panels, keyed by tab and panel ID in this format:
   * `${tab}:${panelId}`
   */
  private panels: Map<string, PanelType> = new Map();

  /**
   * Map of waiters for panel registration.
   */
  private waiters: Map<string, Waiter> = new Map();

  /**
   * A flag indicating if the application is currently switching tabs, which can take some time.
   */
  private _$isLoading = atom(false);
  $isLoading: Atom<boolean> = this._$isLoading;

  /**
   * Separator used to create unique keys for panels. Typo protection.
   */
  KEY_SEPARATOR = ':';

  /**
   * The application API that provides methods to set and get the current app tab and manage panel storage.
   */
  _app: NavigationAppApi | null = null;

  /**
   * Connect to the application to manage tab switching.
   * @param api - The application API that provides methods to set and get the current app tab and manage panel
   *    state storage.
   */
  connectToApp = (api: NavigationAppApi): void => {
    this._app = api;
  };

  /**
   * Disconnect from the application, clearing the tab management functions.
   */
  disconnectFromApp = (): void => {
    this._app = null;
  };

  /**
   * Sets the flag indicating that the navigation is loading and schedules a debounced hide of the loading screen.
   */
  _showFakeLoadingScreen = () => {
    log.debug('Showing fake loading screen for tab switch');
    this._$isLoading.set(true);
    this._hideLoadingScreenDebounced();
  };

  /**
   * Debounced function to hide the loading screen after a delay.
   */
  _hideLoadingScreenDebounced = debounce(() => {
    log.debug('Hiding fake loading screen for tab switch');
    this._$isLoading.set(false);
  }, SWITCH_TABS_FAKE_DELAY_MS);

  /**
   * Switch to a specific app tab.
   *
   * The loading screen will be shown while the tab is switching (and for a little while longer to smooth out the UX).
   *
   * @param tab - The tab to switch to
   * @return True if the switch was successful, false otherwise
   */
  switchToTab = (tab: TabName): boolean => {
    if (!this._app) {
      log.error('No app connected to switch tabs');
      return false;
    }

    if (tab === this._app.activeTab.get()) {
      log.debug(`Already on tab: ${tab}`);
      return true;
    }

    log.debug(`Switching to tab: ${tab}`);
    this._showFakeLoadingScreen();
    this._app.activeTab.set(tab);
    return true;
  };

  /**
   * Registers a panel with the navigation API.
   *
   * @param tab - The tab this panel belongs to
   * @param panelId - Unique identifier for the panel
   * @param panel - The panel instance
   * @returns Cleanup function to unregister the panel
   */
  _registerPanel = <T extends PanelType>(tab: TabName, panelId: string, panel: T): (() => void) => {
    const key = this._getPanelKey(tab, panelId);

    this.panels.set(key, panel);

    // Resolve any pending waiters for this panel, notifying them that the panel is now registered.
    const waiter = this.waiters.get(key);
    if (waiter) {
      if (waiter.timeoutId) {
        clearTimeout(waiter.timeoutId);
      }
      waiter.deferred.resolve();
      this.waiters.delete(key);
    }

    log.debug(`Registered panel ${key}`);

    return () => {
      this.panels.delete(key);
      log.debug(`Unregistered panel ${key}`);
    };
  };

  /**
   * Registers a container (Dockview or Gridview) with the navigation API.
   *
   * This method initializes the container from storage if available, or calls the provided initialize function
   * to set it up from scratch.
   *
   * @param tab - The tab this container belongs to
   * @param id - Unique identifier for the container
   * @param api - The DockviewApi or GridviewApi instance
   * @param initialize - Function to call if the container needs to be initialized from scratch
   */
  registerContainer = (tab: TabName, id: string, api: DockviewApi | GridviewApi, initialize: () => void) => {
    if (!this._app) {
      log.error('App not connected to register view');
      return;
    }

    const key = this._getContainerKey(tab, id);

    const stored = this._app.storage.get(key);
    if (stored) {
      try {
        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        api.fromJSON(stored as any);
        log.debug({ stored: parseify(stored) }, `Restored view ${key} from storage`);
      } catch (error) {
        log.error({ error: parseify(error) }, `Failed to restore view ${key} from storage`);
        this._app.storage.delete(key);
        initialize();
        this._app.storage.set(key, api.toJSON());
      }
    } else {
      initialize();
      log.debug(`Initialized ${key} from scratch`);
      this._app.storage.set(key, api.toJSON());
    }

    for (const panel of api.panels) {
      this._registerPanel(tab, panel.id, panel);
    }

    api.onDidLayoutChange(
      debounce(() => {
        this._app?.storage.set(key, api.toJSON());
      }, 300)
    );

    log.debug(`Registered view ${key}`);
  };

  /**
   * Waits for a panel to be ready.
   *
   * @param tab - The tab the panel belongs to
   * @param panelId - The panel ID to wait for
   * @param timeout - Timeout in milliseconds (default: 2000)
   * @returns Promise that resolves when the panel is ready or rejects if it times out
   *
   * @example
   * ```typescript
   * try {
   *   await navigationApi.waitForPanel('myTab', 'myPanelId');
   *   console.log('Panel is ready');
   * } catch (error) {
   *   console.error('Panel registration timed out:', error);
   * }
   * ```
   */
  waitForPanel = (tab: TabName, panelId: string, timeout = 2000): Promise<void> => {
    const key = this._getPanelKey(tab, panelId);

    // If the panel is already registered, we can resolve immediately.
    if (this.panels.has(key)) {
      return Promise.resolve();
    }

    // If we already have a waiter for this panel, return its promise instead of creating a new one.
    const existing = this.waiters.get(key);
    if (existing) {
      return existing.deferred.promise;
    }

    // We do not have any waiters; create one and set up the timeout.
    const deferred = createDeferredPromise<void>();

    const timeoutId = setTimeout(() => {
      // If the timeout expires, reject the promise and clean up the waiter.
      const waiter = this.waiters.get(key);
      if (waiter) {
        this.waiters.delete(key);
        deferred.reject(new Error(`Panel ${key} registration timed out after ${timeout}ms`));
      }
    }, timeout);

    this.waiters.set(key, { deferred, timeoutId });
    return deferred.promise;
  };

  /**
   * Get the prefix for a tab to create unique keys for panels/containers.
   */
  _getTabPrefix = (tab: TabName): string => {
    return `${tab}${this.KEY_SEPARATOR}`;
  };

  /**
   * Gets a prefix for a panel based on its tab.
   */
  _getPanelPrefix = (tab: TabName): string => {
    return `${this._getTabPrefix(tab)}panel${this.KEY_SEPARATOR}`;
  };

  /**
   * Get the unique key for a panel based on its tab and ID.
   */
  _getPanelKey = (tab: TabName, panelId: string): string => {
    return `${this._getPanelPrefix(tab)}${panelId}`;
  };

  /**
   * Gets a prefix for a container based on its tab.
   */
  _getContainerPrefix = (tab: TabName): string => {
    return `${this._getTabPrefix(tab)}container${this.KEY_SEPARATOR}`;
  };

  /**
   * Get the unique key for a container based on its tab and ID.
   */
  _getContainerKey = (tab: TabName, viewId: string): string => {
    return `${this._getContainerPrefix(tab)}${viewId}`;
  };

  /**
   * Focuses a specific panel in a specific tab.
   *
   * This method does not throw; it returns a Promise that resolves to true if the panel was successfully focused,
   * or false if it failed to focus the panel (e.g., if the panel was not found or the tab switch failed).
   *
   * @param tab - The tab to switch to
   * @param panelId - The panel ID to focus
   * @param timeout - Timeout in milliseconds (default: 2000)
   * @returns Promise that resolves to true if successful, false otherwise
   *
   * @example
   * ```typescript
   * const focused = await navigationApi.focusPanel('myTab', 'myPanelId');
   * if (focused) {
   *   console.log('Panel focused successfully');
   * } else {
   *   console.error('Failed to focus panel');
   * }
   * ```
   */
  focusPanel = async (tab: TabName, panelId: string, timeout = 2000): Promise<boolean> => {
    try {
      this.switchToTab(tab);
      await this.waitForPanel(tab, panelId, timeout);

      const key = this._getPanelKey(tab, panelId);
      const panel = this.panels.get(key);

      if (!panel) {
        log.error(`Panel ${key} not found after waiting`);
        return false;
      }

      // Dockview uses the term "active", but we use "focused" for consistency.
      panel.api.setActive();
      log.debug(`Focused panel ${key}`);

      return true;
    } catch (error) {
      log.error(`Failed to focus panel ${panelId} in tab ${tab}`);
      return false;
    }
  };

  /**
   * Focuses a specific panel in the currently active tab.
   *
   * If the panel does not exist in the active tab, it returns false after a timeout.
   *
   * @param panelId - The panel ID to focus
   * @param timeout - Timeout in milliseconds (default: 2000)
   * @return Promise that resolves to true if the panel was focused, false otherwise
   *
   * @example
   * ```typescript
   * const focused = await navigationApi.focusPanelInActiveTab('myPanelId');
   * if (focused) {
   *   console.log('Panel focused successfully in active tab');
   * } else {
   *   console.error('Failed to focus panel in active tab');
   * }
   */
  focusPanelInActiveTab = (panelId: string, timeout = 2000): Promise<boolean> => {
    const activeTab = this._app?.activeTab.get() ?? null;
    if (!activeTab) {
      log.error('No active tab found');
      return Promise.resolve(false);
    }
    return this.focusPanel(activeTab, panelId, timeout);
  };

  /**
   * Expand a panel to a specified width.
   */
  _expandPanel = (panel: IGridviewPanel, width: number) => {
    panel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: width });
    panel.api.setSize({ width: width });
  };

  /**
   * Collapse a panel by setting its width to 0.
   */
  _collapsePanel = (panel: IGridviewPanel) => {
    panel.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
    panel.api.setSize({ width: 0 });
  };

  /**
   * Get a panel by its tab and ID.
   *
   * This method will not wait for the panel to be registered.
   *
   * @param tab - The tab the panel belongs to
   * @param panelId - The panel ID
   * @returns The panel instance or undefined if not found
   */
  getPanel = (tab: TabName, panelId: string): PanelType | undefined => {
    const key = this._getPanelKey(tab, panelId);
    return this.panels.get(key);
  };

  /**
   * Toggle the left panel in the currently active tab.
   *
   * This method will not wait for the panel to be registered.
   *
   * @returns True if the panel was toggled, false if it was not found or an error occurred
   */
  toggleLeftPanel = (): boolean => {
    const activeTab = this._app?.activeTab.get() ?? null;
    if (!activeTab) {
      log.warn('No active tab found to toggle left panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, LEFT_PANEL_ID);
    if (!leftPanel) {
      log.warn(`Left panel not found in active tab "${activeTab}"`);
      return false;
    }

    if (!(leftPanel instanceof GridviewPanel)) {
      log.error(`Right panels must be instances of GridviewPanel`);
      return false;
    }

    const isCollapsed = leftPanel.width === 0;
    if (isCollapsed) {
      this._expandPanel(leftPanel, LEFT_PANEL_MIN_SIZE_PX);
    } else {
      this._collapsePanel(leftPanel);
    }
    return true;
  };

  /**
   * Toggle the right panel in the currently active tab.
   *
   * This method will not wait for the panel to be registered.
   *
   * @returns True if the panel was toggled, false if it was not found or an error occurred
   */
  toggleRightPanel = (): boolean => {
    const activeTab = this._app?.activeTab.get() ?? null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const rightPanel = this.getPanel(activeTab, RIGHT_PANEL_ID);
    if (!rightPanel) {
      log.warn(`Right panel not found in active tab "${activeTab}"`);
      return false;
    }

    if (!(rightPanel instanceof GridviewPanel)) {
      log.error(`Right panels must be instances of GridviewPanel`);
      return false;
    }

    const isCollapsed = rightPanel.width === 0;
    if (isCollapsed) {
      this._expandPanel(rightPanel, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      this._collapsePanel(rightPanel);
    }
    return true;
  };

  /**
   * Toggle the left and right panels in the currently active tab.
   *
   * This method will not wait for the panels to be registered. If either panel is not found, it will not toggle
   * either panel.
   *
   * @returns True if the panels were toggled, false if they were not found or an error occurred
   */
  toggleLeftAndRightPanels = (): boolean => {
    const activeTab = this._app?.activeTab.get() ?? null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, LEFT_PANEL_ID);
    const rightPanel = this.getPanel(activeTab, RIGHT_PANEL_ID);

    if (!rightPanel || !leftPanel) {
      log.warn(`Right and/or left panel not found in tab "${activeTab}"`);
      return false;
    }

    if (!(leftPanel instanceof GridviewPanel) || !(rightPanel instanceof GridviewPanel)) {
      log.error(`Left and right panels must be instances of GridviewPanel`);
      return false;
    }

    const isLeftCollapsed = leftPanel.width === 0;
    const isRightCollapsed = rightPanel.width === 0;

    if (isLeftCollapsed || isRightCollapsed) {
      this._expandPanel(leftPanel, LEFT_PANEL_MIN_SIZE_PX);
      this._expandPanel(rightPanel, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      this._collapsePanel(leftPanel);
      this._collapsePanel(rightPanel);
    }
    return true;
  };

  /**
   * Reset both left and right panels in the currently active tab to their minimum sizes.
   *
   * This method will not wait for the panels to be registered. If either panel is not found, it will not reset
   * either panel.
   *
   * @returns True if the panels were reset, false if they were not found or an error occurred
   */
  resetLeftAndRightPanels = (): boolean => {
    const activeTab = this._app?.activeTab.get() ?? null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, LEFT_PANEL_ID);
    const rightPanel = this.getPanel(activeTab, RIGHT_PANEL_ID);

    if (!rightPanel || !leftPanel) {
      log.warn(`Right and/or left panel not found in tab "${activeTab}"`);
      return false;
    }

    if (!(leftPanel instanceof GridviewPanel) || !(rightPanel instanceof GridviewPanel)) {
      log.error(`Left and right panels must be instances of GridviewPanel`);
      return false;
    }

    leftPanel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: LEFT_PANEL_MIN_SIZE_PX });
    leftPanel.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });

    rightPanel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: RIGHT_PANEL_MIN_SIZE_PX });
    rightPanel.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });

    return true;
  };

  /**
   * Check if a panel is registered.
   * @param tab - The tab the panel belongs to
   * @param panelId - The panel ID to check
   * @returns True if the panel is registered
   */
  isPanelRegistered = (tab: TabName, panelId: string): boolean => {
    const key = this._getPanelKey(tab, panelId);
    return this.panels.has(key);
  };

  /**
   * Get all registered panels for a tab.
   * @param tab - The tab to get panels for
   * @returns Array of panel IDs
   */
  getRegisteredPanels = (tab: TabName): string[] => {
    const prefix = this._getPanelPrefix(tab);
    return Array.from(this.panels.keys())
      .filter((key) => key.startsWith(prefix))
      .map((key) => key.substring(prefix.length));
  };

  /**
   * Unregister all panels for a tab. Any pending waiters for these panels will be rejected.
   * @param tab - The tab to unregister panels for
   */
  unregisterTab = (tab: TabName): void => {
    const prefix = this._getPanelPrefix(tab);
    const keysToDelete = Array.from(this.panels.keys()).filter((key) => key.startsWith(prefix));

    for (const key of keysToDelete) {
      this.panels.delete(key);
    }

    const promiseKeysToDelete = Array.from(this.waiters.keys()).filter((key) => key.startsWith(prefix));
    for (const key of promiseKeysToDelete) {
      const waiter = this.waiters.get(key);
      if (waiter) {
        // Clear timeout before rejecting to prevent multiple rejections
        if (waiter.timeoutId) {
          clearTimeout(waiter.timeoutId);
        }
        waiter.deferred.reject(new Error(`Panel registration cancelled - tab ${tab} was unregistered`));
      }
      this.waiters.delete(key);
    }

    log.debug(`Unregistered all panels for tab ${tab}`);
  };
}

export const navigationApi = new NavigationApi();
