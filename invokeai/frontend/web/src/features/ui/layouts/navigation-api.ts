import { logger } from 'app/logging/logger';
import { createDeferredPromise, type Deferred } from 'common/util/createDeferredPromise';
import { GridviewPanel, type IDockviewPanel, type IGridviewPanel } from 'dockview';
import type { TabName } from 'features/ui/store/uiTypes';

import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from './shared';

const log = logger('system');

type PanelType = IGridviewPanel | IDockviewPanel;

type Waiter = {
  deferred: Deferred<void>;
  timeoutId: ReturnType<typeof setTimeout> | null;
};

const PANEL_ENABLED_TABS: TabName[] = ['canvas', 'generate', 'workflows', 'queue'];

export class NavigationApi {
  private panels: Map<string, PanelType> = new Map();
  private waiters: Map<string, Waiter> = new Map();

  setAppTab: ((tab: TabName) => void) | null = null;
  getAppTab: (() => TabName) | null = null;

  connectToApp = (arg: { setAppTab: (tab: TabName) => void; getAppTab: () => TabName }): void => {
    const { setAppTab, getAppTab } = arg;
    this.setAppTab = setAppTab;
    this.getAppTab = getAppTab;
  };

  disconnectFromApp = (): void => {
    this.setAppTab = null;
    this.getAppTab = null;
  };

  /**
   * Register a panel with a unique ID
   * @param tab - The tab this panel belongs to
   * @param panelId - Unique identifier for the panel
   * @param panel - The panel instance
   * @returns Cleanup function to unregister the panel
   */
  registerPanel = (tab: TabName, panelId: string, panel: PanelType): (() => void) => {
    const key = `${tab}:${panelId}`;

    this.panels.set(key, panel);

    // Resolve any waiting promises
    const waiter = this.waiters.get(key);
    if (waiter) {
      if (waiter.timeoutId) {
        // Clear the timeout if it exists
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
   * Wait for a panel to be ready
   * @param tab - The tab the panel belongs to
   * @param panelId - The panel ID to wait for
   * @param timeout - Timeout in milliseconds (default: 2000)
   * @returns Promise that resolves when the panel is ready
   */
  waitForPanel = (tab: TabName, panelId: string, timeout = 2000): Promise<void> => {
    if (!PANEL_ENABLED_TABS.includes(tab)) {
      log.error(`Tab ${tab} is not enabled for panel registration`);
      return Promise.reject(new Error(`Tab ${tab} is not enabled for panel registration`));
    }

    const key = `${tab}:${panelId}`;

    if (this.panels.has(key)) {
      return Promise.resolve();
    }

    // Check if we already have a promise for this panel
    const existing = this.waiters.get(key);

    if (existing) {
      return existing.deferred.promise;
    }

    const deferred = createDeferredPromise<void>();

    const timeoutId = setTimeout(() => {
      // Only reject if this deferred is still waiting
      const waiter = this.waiters.get(key);
      if (waiter) {
        this.waiters.delete(key);
        deferred.reject(new Error(`Panel ${key} registration timed out after ${timeout}ms`));
      }
    }, timeout);

    this.waiters.set(key, { deferred, timeoutId });
    return deferred.promise;
  };

  getPanelKey = (tab: TabName, panelId: string): string => {
    return `${tab}:${panelId}`;
  };

  /**
   * Focus a specific panel in a specific tab
   * @param tab - The tab to switch to
   * @param panelId - The panel ID to focus
   * @returns Promise that resolves to true if successful, false otherwise
   */
  focusPanel = async (tab: TabName, panelId: string): Promise<boolean> => {
    if (!PANEL_ENABLED_TABS.includes(tab)) {
      log.error(`Tab ${tab} is not enabled for panel registration`);
      return Promise.resolve(false);
    }

    try {
      // Switch to the target tab if needed
      if (this.setAppTab && this.getAppTab && this.getAppTab() !== tab) {
        this.setAppTab(tab);
      }

      // Wait for the panel to be ready
      await this.waitForPanel(tab, panelId);

      const key = this.getPanelKey(tab, panelId);
      const panel = this.panels.get(key);

      if (!panel) {
        log.error(`Panel ${key} not found after waiting`);
        return false;
      }

      // Focus the panel
      panel.api.setActive();
      log.debug(`Focused panel ${key}`);

      return true;
    } catch (error) {
      log.error(`Failed to focus panel ${panelId} in tab ${tab}`);
      return false;
    }
  };

  focusPanelInActiveTab = (panelId: string): Promise<boolean> => {
    const activeTab = this.getAppTab ? this.getAppTab() : null;
    if (!activeTab) {
      log.error('No active tab found');
      return Promise.resolve(false);
    }
    return this.focusPanel(activeTab, panelId);
  };

  expandPanel = (panel: IGridviewPanel, width: number) => {
    panel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: width });
    panel.api.setSize({ width: width });
  };

  collapsePanel = (panel: IGridviewPanel) => {
    panel.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
    panel.api.setSize({ width: 0 });
  };

  getPanel = (tab: TabName, panelId: string): PanelType | undefined => {
    if (!PANEL_ENABLED_TABS.includes(tab)) {
      log.warn(`Tab ${tab} is not enabled for panel registration`);
      return undefined;
    }
    const key = this.getPanelKey(tab, panelId);
    return this.panels.get(key);
  };

  toggleLeftPanel = (): boolean => {
    const activeTab = this.getAppTab ? this.getAppTab() : null;
    if (!activeTab) {
      log.warn('No active tab found to toggle left panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, 'left');
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
      this.expandPanel(leftPanel, LEFT_PANEL_MIN_SIZE_PX);
    } else {
      this.collapsePanel(leftPanel);
    }
    return true;
  };

  toggleRightPanel = (): boolean => {
    const activeTab = this.getAppTab ? this.getAppTab() : null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const rightPanel = this.getPanel(activeTab, 'right');
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
      this.expandPanel(rightPanel, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      this.collapsePanel(rightPanel);
    }
    return true;
  };

  toggleLeftAndRightPanels = (): boolean => {
    const activeTab = this.getAppTab ? this.getAppTab() : null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, 'left');
    const rightPanel = this.getPanel(activeTab, 'right');

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
      this.expandPanel(leftPanel, LEFT_PANEL_MIN_SIZE_PX);
      this.expandPanel(rightPanel, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      this.collapsePanel(leftPanel);
      this.collapsePanel(rightPanel);
    }
    return true;
  };

  /**
   * Reset panels in a specific tab (expand both left and right)
   */
  resetLeftAndRightPanels = (): boolean => {
    const activeTab = this.getAppTab ? this.getAppTab() : null;
    if (!activeTab) {
      log.warn('No active tab found to toggle right panel');
      return false;
    }
    const leftPanel = this.getPanel(activeTab, 'left');
    const rightPanel = this.getPanel(activeTab, 'right');

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
   * Check if a panel is registered
   * @param tab - The tab the panel belongs to
   * @param panelId - The panel ID to check
   * @returns True if the panel is registered
   */
  isPanelRegistered = (tab: TabName, panelId: string): boolean => {
    const key = `${tab}:${panelId}`;
    return this.panels.has(key);
  };

  /**
   * Get all registered panels for a tab
   * @param tab - The tab to get panels for
   * @returns Array of panel IDs
   */
  getRegisteredPanels = (tab: TabName): string[] => {
    const prefix = `${tab}:`;
    return Array.from(this.panels.keys())
      .filter((key) => key.startsWith(prefix))
      .map((key) => key.substring(prefix.length));
  };

  /**
   * Unregister all panels for a tab
   * @param tab - The tab to unregister panels for
   */
  unregisterTab = (tab: TabName): void => {
    const prefix = `${tab}:`;
    const keysToDelete = Array.from(this.panels.keys()).filter((key) => key.startsWith(prefix));

    for (const key of keysToDelete) {
      this.panels.delete(key);
    }

    // Clean up any pending promises by rejecting them
    const promiseKeysToDelete = Array.from(this.waiters.keys()).filter((key) => key.startsWith(prefix));
    for (const key of promiseKeysToDelete) {
      const waiter = this.waiters.get(key);
      if (waiter) {
        // Clear timeout before rejecting
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
