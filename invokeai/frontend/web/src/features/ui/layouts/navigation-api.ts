import { logger } from 'app/logging/logger';
import { createDeferredPromise, type Deferred } from 'common/util/createDeferredPromise';
import { GridviewPanel, type IDockviewPanel, type IGridviewPanel } from 'dockview';
import type { DockviewPanelState, GridviewPanelState,TabName  } from 'features/ui/store/uiTypes';
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
 * Callbacks for getting and setting panel state from the Redux store.
 */
type PanelStateCallbacks = {
  getGridviewPanelState: (id: string) => GridviewPanelState | undefined;
  setGridviewPanelState: (id: string, state: GridviewPanelState) => void;
  getDockviewPanelState: (id: string) => DockviewPanelState | undefined;
  setDockviewPanelState: (id: string, state: DockviewPanelState) => void;
  getActiveTabMainPanel?: (tab: TabName) => string | undefined;
  setActiveTabMainPanel?: (tab: TabName, panel: string) => void;
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
  $isSwitchingTabs = atom(false);
  /**
   * The timeout used to add a short additional delay when switching tabs.
   *
   * The time it takes to switch tabs varies depending on the tab, and sometimes it is very fast, resulting in a flicker
   * of the loading screen. This timeout is used to artificially extend the time the loading screen is shown.
   */
  switchingTabsTimeout: ReturnType<typeof setTimeout> | null = null;

  /**
   * Separator used to create unique keys for panels. Typo protection.
   */
  KEY_SEPARATOR = ':';

  /**
   * Private imperative method to set the current app tab.
   */
  _setAppTab: ((tab: TabName) => void) | null = null;

  /**
   * Private imperative method to get the current app tab.
   */
  _getAppTab: (() => TabName) | null = null;

  /**
   * Panel state callbacks for interacting with the Redux store.
   */
  private _panelStateCallbacks: PanelStateCallbacks | null = null;

  /**
   * Connect to the application to manage tab switching and panel state.
   * @param arg.setAppTab - Function to set the current app tab
   * @param arg.getAppTab - Function to get the current app tab
   * @param arg.panelStateCallbacks - Optional callbacks for panel state persistence
   */
  connectToApp = (arg: { 
    setAppTab: (tab: TabName) => void; 
    getAppTab: () => TabName;
    panelStateCallbacks?: PanelStateCallbacks;
  }): void => {
    const { setAppTab, getAppTab, panelStateCallbacks } = arg;
    this._setAppTab = setAppTab;
    this._getAppTab = getAppTab;
    this._panelStateCallbacks = panelStateCallbacks || null;
  };

  /**
   * Disconnect from the application, clearing the tab management functions and panel state callbacks.
   */
  disconnectFromApp = (): void => {
    this._setAppTab = null;
    this._getAppTab = null;
    this._panelStateCallbacks = null;
  };

  /**
   * Switch to a specific app tab.
   *
   * The loading screen will be shown while the tab is switching.
   *
   * @param tab - The tab to switch to
   * @return True if the switch was successful, false otherwise
   */
  switchToTab = (tab: TabName): boolean => {
    if (this.switchingTabsTimeout !== null) {
      clearTimeout(this.switchingTabsTimeout);
      this.switchingTabsTimeout = null;
    }
    if (tab === this._getAppTab?.()) {
      return true;
    }
    this.$isSwitchingTabs.set(true);
    log.debug(`Switching to tab: ${tab}`);
    if (this._setAppTab) {
      this._setAppTab(tab);
      return true;
    } else {
      log.error('No setAppTab function available to switch tabs');
      return false;
    }
  };

  /**
   * Callback for when a tab is ready after switching.
   *
   * Hides the loading screen after a short delay.
   */
  onTabReady = (tab: TabName): void => {
    this.switchingTabsTimeout = setTimeout(() => {
      this.$isSwitchingTabs.set(false);
      log.debug(`Tab ${tab} ready`);
    }, SWITCH_TABS_FAKE_DELAY_MS);
  };

  /**
   * Rehydrates panel state from the Redux store if available.
   */
  private _rehydratePanelState = (tab: TabName, panelId: string, panel: PanelType): void => {
    if (!this._panelStateCallbacks) {
      return;
    }

    const key = this._getPanelKey(tab, panelId);

    if (panel instanceof GridviewPanel) {
      const savedState = this._panelStateCallbacks.getGridviewPanelState(key);
      if (savedState) {
        if (savedState.width !== undefined || savedState.height !== undefined) {
          const size: { width?: number; height?: number } = {};
          if (savedState.width !== undefined) {
            size.width = savedState.width;
          }
          if (savedState.height !== undefined) {
            size.height = savedState.height;
          }
          panel.api.setSize(size);
          log.debug(`Rehydrated Gridview panel ${key} with state`);
        }
      }
    } else {
      // Dockview panel
      const savedState = this._panelStateCallbacks.getDockviewPanelState(key);
      if (savedState && savedState.isActive) {
        panel.api.setActive();
        log.debug(`Rehydrated Dockview panel ${key} with active state`);
      }
    }

    // Special handling for main panels to restore active state
    if (this._panelStateCallbacks.getActiveTabMainPanel) {
      const activeMainPanel = this._panelStateCallbacks.getActiveTabMainPanel(tab);
      if (activeMainPanel === panelId && panel instanceof GridviewPanel === false) {
        // This is a dockview panel and it should be active
        panel.api.setActive();
        log.debug(`Rehydrated ${tab} main panel ${panelId} as active`);
      }
    }
  };

  /**
   * Sets up listeners to persist panel state changes.
   */
  private _setupPanelStateListeners = (tab: TabName, panelId: string, panel: PanelType): (() => void) => {
    if (!this._panelStateCallbacks) {
      return () => {};
    }

    const key = this._getPanelKey(tab, panelId);
    const disposables: (() => void)[] = [];

    if (panel instanceof GridviewPanel) {
      const onDimensionsChange = panel.api.onDidDimensionsChange(() => {
        const { width, height } = panel.api;
        this._panelStateCallbacks!.setGridviewPanelState(key, { width, height });
        log.debug(`Persisted Gridview panel ${key} dimensions`);
      });
      disposables.push(() => onDimensionsChange.dispose());
    } else {
      // Dockview panel
      const onActiveChange = panel.api.onDidActiveChange((event) => {
        const isActive = event.isActive;
        this._panelStateCallbacks!.setDockviewPanelState(key, { isActive });
        log.debug(`Persisted Dockview panel ${key} active state`);
        
        // Special handling for main panels to persist active state
        if (this._panelStateCallbacks?.setActiveTabMainPanel && isActive) {
          this._panelStateCallbacks.setActiveTabMainPanel(tab, panelId);
          log.debug(`Persisted ${tab} main panel ${panelId} as active`);
        }
      });
      disposables.push(() => onActiveChange.dispose());
    }

    return () => {
      disposables.forEach((dispose) => dispose());
    };
  };

  /**
   * Registers a panel with the navigation API.
   *
   * @param tab - The tab this panel belongs to
   * @param panelId - Unique identifier for the panel
   * @param panel - The panel instance
   * @returns Cleanup function to unregister the panel
   */
  registerPanel = (tab: TabName, panelId: string, panel: PanelType): (() => void) => {
    const key = this._getPanelKey(tab, panelId);

    this.panels.set(key, panel);

    // Rehydrate panel state from Redux store
    this._rehydratePanelState(tab, panelId, panel);

    // Set up listeners to persist state changes
    const cleanupListeners = this._setupPanelStateListeners(tab, panelId, panel);

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
      cleanupListeners();
      this.panels.delete(key);
      log.debug(`Unregistered panel ${key}`);
    };
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
   * Get the prefix for a tab to create unique keys for panels.
   */
  _getTabPrefix = (tab: TabName): string => {
    return `${tab}${this.KEY_SEPARATOR}`;
  };

  /**
   * Get the unique key for a panel based on its tab and ID.
   */
  _getPanelKey = (tab: TabName, panelId: string): string => {
    return `${this._getTabPrefix(tab)}${panelId}`;
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
    const activeTab = this._getAppTab ? this._getAppTab() : null;
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
    const activeTab = this._getAppTab ? this._getAppTab() : null;
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

    const isCollapsed = leftPanel.maximumWidth === 0;
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
    const activeTab = this._getAppTab ? this._getAppTab() : null;
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

    const isCollapsed = rightPanel.maximumWidth === 0;
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
    const activeTab = this._getAppTab ? this._getAppTab() : null;
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

    const isLeftCollapsed = leftPanel.maximumWidth === 0;
    const isRightCollapsed = rightPanel.maximumWidth === 0;

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
    const activeTab = this._getAppTab ? this._getAppTab() : null;
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
    const prefix = this._getTabPrefix(tab);
    return Array.from(this.panels.keys())
      .filter((key) => key.startsWith(prefix))
      .map((key) => key.substring(prefix.length));
  };

  /**
   * Unregister all panels for a tab. Any pending waiters for these panels will be rejected.
   * @param tab - The tab to unregister panels for
   */
  unregisterTab = (tab: TabName): void => {
    const prefix = this._getTabPrefix(tab);
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

  /**
   * Get the active main panel for a specific tab from the persisted state.
   * @param tab - The tab to get the active main panel for
   * @returns The active panel ID or undefined if not set
   */
  getActiveTabMainPanel = (tab: TabName): string | undefined => {
    return this._panelStateCallbacks?.getActiveTabMainPanel?.(tab);
  };

  /**
   * Get the active canvas main panel from the persisted state.
   * @returns The active panel ID or undefined if not set
   */
  getActiveCanvasMainPanel = (): string | undefined => {
    return this.getActiveTabMainPanel('canvas');
  };

  /**
   * Get the persisted state for a gridview panel.
   * @param id - The panel ID (including tab prefix)
   * @returns The persisted panel state or undefined if not found
   */
  getGridviewPanelState = (id: string): GridviewPanelState | undefined => {
    return this._panelStateCallbacks?.getGridviewPanelState?.(id);
  };
}

export const navigationApi = new NavigationApi();
