import { logger } from 'app/logging/logger';
import type { DockviewApi, GridviewApi, IGridviewPanel } from 'dockview';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/layouts/shared';
import type { TabName } from 'features/ui/store/uiTypes';

const log = logger('system');

export type TabPanelApis = {
  root: Readonly<GridviewApi> | null;
  left: Readonly<GridviewApi> | null;
  main: Readonly<DockviewApi> | null;
  right: Readonly<GridviewApi> | null;
};

const getInitialTabPanelApis = (): TabPanelApis => ({
  root: null,
  left: null,
  main: null,
  right: null,
});

type AppTabApi = {
  setTab: (tabName: TabName) => void;
  getTab: () => TabName;
};

/**
 * Global registry for all panel APIs across all tabs
 */
export class PanelRegistry {
  private tabPanelApis = new Map<TabName, TabPanelApis>();

  tabApi: AppTabApi | null = null;

  /**
   * Set the Redux store reference for tab switching
   */
  setTabApi(tabApi: AppTabApi): void {
    this.tabApi = tabApi;
  }

  /**
   * Register panel APIs for a specific tab
   */
  registerPanel<T extends keyof TabPanelApis>(tab: TabName, panel: T, api: NonNullable<TabPanelApis[T]>): void {
    const current = this.tabPanelApis.get(tab) ?? getInitialTabPanelApis();
    const apis: TabPanelApis = {
      ...current,
      [panel]: api,
    };
    this.tabPanelApis.set(tab, apis);
  }

  /**
   * Unregister panel APIs for a specific tab
   */
  unregisterPanel<T extends keyof TabPanelApis>(tab: TabName, panel: T): void {
    const current = this.tabPanelApis.get(tab);
    if (!current) {
      return;
    }
    const apis: TabPanelApis = {
      ...current,
      [panel]: null,
    };
    this.tabPanelApis.set(tab, apis);
  }

  /**
   * Unregister panel APIs for a specific tab
   */
  unregisterTab(tabName: TabName): void {
    this.tabPanelApis.delete(tabName);
  }

  /**
   * Get panel APIs for a specific tab
   */
  getTabPanelApis(tabName: TabName): TabPanelApis | null {
    return this.tabPanelApis.get(tabName) || null;
  }

  /**
   * Get panel APIs for the currently active tab
   */
  getActiveTabPanelApis(): TabPanelApis | null {
    if (!this.tabApi) {
      return null;
    }

    const activeTab = this.tabApi.getTab();
    return this.getTabPanelApis(activeTab);
  }

  /**
   * Get all registered tab names
   */
  getRegisteredTabs(): TabName[] {
    return Array.from(this.tabPanelApis.keys());
  }

  /**
   * Check if a tab is registered
   */
  isTabRegistered(tabName: TabName): boolean {
    return this.tabPanelApis.has(tabName);
  }

  /**
   * Switch to a specific tab
   */
  private switchToTab(tabName: TabName): boolean {
    if (!this.tabApi) {
      log.warn(`Cannot switch to tab "${tabName}": no store reference`);
      return false;
    }

    this.tabApi.setTab(tabName);
    return true;
  }

  /**
   * Focus a specific panel in a specific tab
   * Automatically switches to the target tab if specified
   */
  focusPanelInTab(tabName: TabName, panelId: string, switchTab = true): boolean {
    const apis = this.getTabPanelApis(tabName);
    if (!apis) {
      log.warn(`Tab "${tabName}" not registered`);
      return false;
    }

    if (switchTab) {
      // Switch to target tab first
      if (!this.switchToTab(tabName)) {
        return false;
      }
    }

    // Try to focus in main panel (dockview) first
    if (apis.main) {
      const panel = apis.main.getPanel(panelId);
      if (panel) {
        panel.api.setActive();
        return true;
      }
    }

    // Try left panel
    if (apis.left) {
      const panel = apis.left.getPanel(panelId);
      if (panel) {
        panel.api.setActive();
        return true;
      }
    }

    // Try right panel
    if (apis.right) {
      const panel = apis.right.getPanel(panelId);
      if (panel) {
        panel.api.setActive();
        return true;
      }
    }

    log.warn(`Panel "${panelId}" not found in tab "${tabName}"`);
    return false;
  }

  /**
   * Focus a panel in the currently active tab
   */
  focusPanelInActiveTab(panelId: string): boolean {
    if (!this.tabApi) {
      return false;
    }

    const activeTab = this.tabApi.getTab();
    return this.focusPanelInTab(activeTab, panelId);
  }

  expandPanel(panel: IGridviewPanel, width: number) {
    panel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: width });
    panel.api.setSize({ width: width });
  }

  collapsePanel(panel: IGridviewPanel) {
    panel.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
    panel.api.setSize({ width: 0 });
  }

  /**
   * Toggle the left panel in a specific tab
   */
  toggleLeftPanelInTab(tabName: TabName): boolean {
    const apis = this.getTabPanelApis(tabName);
    if (!apis?.root) {
      log.warn(`Root panel API not available for tab "${tabName}"`);
      return false;
    }

    if (!this.switchToTab(tabName)) {
      return false;
    }

    const leftPanel = apis.root.getPanel('left');
    if (!leftPanel) {
      log.warn(`Left panel not found in tab "${tabName}"`);
      return false;
    }

    const isCollapsed = leftPanel.maximumWidth === 0;
    if (isCollapsed) {
      this.expandPanel(leftPanel, LEFT_PANEL_MIN_SIZE_PX);
    } else {
      this.collapsePanel(leftPanel);
    }
    return true;
  }

  /**
   * Toggle the right panel in a specific tab
   */
  toggleRightPanelInTab(tabName: TabName): boolean {
    const apis = this.getTabPanelApis(tabName);
    if (!apis?.root) {
      log.warn(`Root panel API not available for tab "${tabName}"`);
      return false;
    }

    if (!this.switchToTab(tabName)) {
      return false;
    }

    const rightPanel = apis.root.getPanel('right');
    if (!rightPanel) {
      log.warn(`Right panel not found in tab "${tabName}"`);
      return false;
    }

    const isCollapsed = rightPanel.maximumWidth === 0;
    if (isCollapsed) {
      this.expandPanel(rightPanel, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      this.collapsePanel(rightPanel);
    }
    return true;
  }

  toggleBothPanelsInTab(tabName: TabName): boolean {
    const apis = this.getTabPanelApis(tabName);
    if (!apis?.root) {
      log.warn(`Root panel API not available for tab "${tabName}"`);
      return false;
    }

    const rightPanel = apis.root.getPanel('right');
    const leftPanel = apis.root.getPanel('left');
    if (!rightPanel || !leftPanel) {
      log.warn(`Right and/or left panel not found in tab "${tabName}"`);
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
  }

  /**
   * Toggle the left panel in the currently active tab
   */
  toggleLeftPanelInActiveTab(): boolean {
    if (!this.tabApi) {
      return false;
    }

    const activeTab = this.tabApi.getTab();
    return this.toggleLeftPanelInTab(activeTab);
  }

  /**
   * Toggle the right panel in the currently active tab
   */
  toggleRightPanelInActiveTab(): boolean {
    if (!this.tabApi) {
      return false;
    }

    const activeTab = this.tabApi.getTab();
    return this.toggleRightPanelInTab(activeTab);
  }

  /**
   * Reset panels in a specific tab (expand both left and right)
   */
  resetPanelsInTab(tabName: TabName): boolean {
    const apis = this.getTabPanelApis(tabName);
    if (!apis?.root) {
      log.warn(`Root panel API not available for tab "${tabName}"`);
      return false;
    }

    if (!this.switchToTab(tabName)) {
      return false;
    }

    const rootApi = apis.root as GridviewApi;
    const leftPanel = rootApi.getPanel('left');
    const rightPanel = rootApi.getPanel('right');

    if (leftPanel) {
      leftPanel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: LEFT_PANEL_MIN_SIZE_PX });
      leftPanel.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
    }

    if (rightPanel) {
      rightPanel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: RIGHT_PANEL_MIN_SIZE_PX });
      rightPanel.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
    }

    return true;
  }

  /**
   * Reset panels in the currently active tab
   */
  resetPanelsInActiveTab(): boolean {
    if (!this.tabApi) {
      return false;
    }

    const activeTab = this.tabApi.getTab();
    return this.resetPanelsInTab(activeTab);
  }
}

// Global singleton instance
export const panelRegistry = new PanelRegistry();
