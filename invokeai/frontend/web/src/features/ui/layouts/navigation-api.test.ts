import type { DockviewApi, GridviewApi } from 'dockview';
import { DockviewPanel, GridviewPanel } from 'dockview';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { NavigationAppApi } from './navigation-api';
import { NavigationApi } from './navigation-api';
import {
  LAUNCHPAD_PANEL_ID,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  SWITCH_TABS_FAKE_DELAY_MS,
  WORKSPACE_PANEL_ID,
} from './shared';

// Mock the logger
vi.mock('app/logging/logger', () => ({
  logger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

vi.mock('dockview', async () => {
  const actual = await vi.importActual('dockview');

  // Mock GridviewPanel class for instanceof checks
  class MockGridviewPanel {
    maximumWidth?: number;
    minimumWidth?: number;
    width?: number;
    api = {
      setActive: vi.fn(),
      setConstraints: vi.fn(),
      setSize: vi.fn(),
      onDidDimensionsChange: vi.fn(() => ({ dispose: vi.fn() })),
    };

    constructor(config: { maximumWidth?: number; minimumWidth?: number; width?: number } = {}) {
      this.maximumWidth = config.maximumWidth;
      this.minimumWidth = config.minimumWidth;
      this.width = config.width;
    }
  }

  // Mock GridviewPanel class for instanceof checks
  class MockDockviewPanel {
    api = {
      setActive: vi.fn(),
      setConstraints: vi.fn(),
      setSize: vi.fn(),
      onDidActiveChange: vi.fn(() => ({ dispose: vi.fn() })),
    };
  }

  return {
    ...actual,
    GridviewPanel: MockGridviewPanel,
    DockviewPanel: MockDockviewPanel,
  };
});

// Mock panel with setActive method
const createMockPanel = (config: { maximumWidth?: number; minimumWidth?: number; width?: number } = {}) => {
  /* @ts-expect-error we are mocking GridviewPanel to be a concrete class */
  return new GridviewPanel(config);
};

const createMockDockPanel = () => {
  /* @ts-expect-error we are mocking GridviewPanel to be a concrete class */
  return new DockviewPanel();
};

describe('AppNavigationApi', () => {
  let navigationApi: NavigationApi;
  let mockSetAppTab: ReturnType<typeof vi.fn>;
  let mockGetAppTab: ReturnType<typeof vi.fn>;
  let mockSetStorage: ReturnType<typeof vi.fn>;
  let mockGetStorage: ReturnType<typeof vi.fn>;
  let mockDeleteStorage: ReturnType<typeof vi.fn>;
  let mockAppApi: NavigationAppApi;

  beforeEach(() => {
    navigationApi = new NavigationApi();
    mockSetAppTab = vi.fn();
    mockGetAppTab = vi.fn();
    mockSetStorage = vi.fn();
    mockGetStorage = vi.fn();
    mockDeleteStorage = vi.fn();
    mockAppApi = {
      activeTab: {
        set: mockSetAppTab,
        get: mockGetAppTab,
      },
      storage: {
        set: mockSetStorage,
        get: mockGetStorage,
        delete: mockDeleteStorage,
      },
    };
  });

  afterEach(() => {
    // Clean up all panels and pending promises to prevent unhandled rejections
    navigationApi.unregisterTab('generate');
    navigationApi.unregisterTab('canvas');
    navigationApi.unregisterTab('upscaling');
    navigationApi.unregisterTab('workflows');
  });

  describe('Basic Connection', () => {
    it('should connect to app', () => {
      navigationApi.connectToApp(mockAppApi);

      expect(navigationApi._app).not.toBeNull();
      expect(navigationApi._app?.activeTab.set).toBe(mockSetAppTab);
      expect(navigationApi._app?.activeTab.get).toBe(mockGetAppTab);
      expect(navigationApi._app?.storage.set).toBe(mockSetStorage);
      expect(navigationApi._app?.storage.get).toBe(mockGetStorage);
      expect(navigationApi._app?.storage.delete).toBe(mockDeleteStorage);
    });

    it('should disconnect from app', () => {
      navigationApi.connectToApp(mockAppApi);
      navigationApi.disconnectFromApp();

      expect(navigationApi._app).toBeNull();
    });
  });

  describe('Tab Switching', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.clearAllTimers();
      vi.useRealTimers();
    });

    it('should switch tabs', () => {
      navigationApi.connectToApp(mockAppApi);
      navigationApi.switchToTab('canvas');
      expect(mockSetAppTab).toHaveBeenCalledWith('canvas');
    });

    it('should not set the tab if it is already on that tab', () => {
      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('canvas');

      navigationApi.switchToTab('canvas');
      expect(mockSetAppTab).not.toHaveBeenCalled();
    });

    it('should set the $isLoading atom when switching', () => {
      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('generate');

      navigationApi.switchToTab('canvas');
      expect(navigationApi.$isLoading.get()).toBe(true);
    });

    it('should unset the $isLoading atom after a fake delay', () => {
      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('generate');

      navigationApi.switchToTab('canvas');
      expect(navigationApi.$isLoading.get()).toBe(true);

      vi.advanceTimersByTime(SWITCH_TABS_FAKE_DELAY_MS);
      expect(navigationApi.$isLoading.get()).toBe(false);
    });

    it('should handle rapid tab changes', () => {
      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('generate');

      navigationApi.switchToTab('canvas');
      expect(navigationApi.$isLoading.get()).toBe(true);

      navigationApi.switchToTab('generate');
      expect(navigationApi.$isLoading.get()).toBe(true);

      vi.advanceTimersByTime(SWITCH_TABS_FAKE_DELAY_MS / 5);

      navigationApi.switchToTab('canvas');
      expect(navigationApi.$isLoading.get()).toBe(true);

      vi.advanceTimersByTime(SWITCH_TABS_FAKE_DELAY_MS / 5);

      navigationApi.switchToTab('generate');
      expect(navigationApi.$isLoading.get()).toBe(true);

      vi.advanceTimersByTime(SWITCH_TABS_FAKE_DELAY_MS / 5);

      navigationApi.switchToTab('canvas');
      expect(navigationApi.$isLoading.get()).toBe(true);

      vi.advanceTimersByTime(SWITCH_TABS_FAKE_DELAY_MS);

      expect(navigationApi.$isLoading.get()).toBe(false);
    });

    it('should not switch tabs if the app is not connected', () => {
      navigationApi.switchToTab('canvas');
      expect(mockSetAppTab).not.toHaveBeenCalled();
    });
  });

  describe('Panel Registration', () => {
    it('should register and unregister panels', () => {
      const mockPanel = createMockPanel();
      const unregister = navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      expect(typeof unregister).toBe('function');
      expect(navigationApi.isPanelRegistered('generate', SETTINGS_PANEL_ID)).toBe(true);

      // Unregister
      unregister();
      expect(navigationApi.isPanelRegistered('generate', SETTINGS_PANEL_ID)).toBe(false);
    });

    it('should resolve waiting promises when panel is registered', async () => {
      const mockPanel = createMockPanel();

      // Start waiting for registration
      const waitPromise = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID);

      // Register the panel
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      // Wait should resolve
      await expect(waitPromise).resolves.toBeUndefined();
    });

    it('should handle multiple panels per tab', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockDockPanel();

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel1);
      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockPanel2);

      expect(navigationApi.isPanelRegistered('generate', SETTINGS_PANEL_ID)).toBe(true);
      expect(navigationApi.isPanelRegistered('generate', LAUNCHPAD_PANEL_ID)).toBe(true);

      const registeredPanels = navigationApi.getRegisteredPanels('generate');
      expect(registeredPanels).toContain(SETTINGS_PANEL_ID);
      expect(registeredPanels).toContain(LAUNCHPAD_PANEL_ID);
    });

    it('should handle panels across different tabs', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockPanel();

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel1);
      navigationApi._registerPanel('canvas', SETTINGS_PANEL_ID, mockPanel2);

      expect(navigationApi.isPanelRegistered('generate', SETTINGS_PANEL_ID)).toBe(true);
      expect(navigationApi.isPanelRegistered('canvas', SETTINGS_PANEL_ID)).toBe(true);

      // Same panel ID in different tabs should be separate
      expect(navigationApi.getRegisteredPanels('generate')).toEqual([SETTINGS_PANEL_ID]);
      expect(navigationApi.getRegisteredPanels('canvas')).toEqual([SETTINGS_PANEL_ID]);
    });
  });

  describe('Panel Focus', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should focus panel in already registered tab', async () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(true);
      expect(mockSetAppTab).not.toHaveBeenCalled();
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should switch tab before focusing panel', async () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('canvas'); // Currently on different tab

      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should wait for panel registration before focusing', async () => {
      const mockPanel = createMockPanel();
      mockGetAppTab.mockReturnValue('generate');

      // Start focus operation before panel is registered
      const focusPromise = navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      // Register panel after a short delay
      setTimeout(() => {
        navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      }, 100);

      const result = await focusPromise;

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should focus different panel types', async () => {
      const mockGridPanel = createMockPanel();
      const mockDockPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockGridPanel);
      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockDockPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Test gridview panel
      const result1 = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);
      expect(result1).toBe(true);
      expect(mockGridPanel.api.setActive).toHaveBeenCalledOnce();

      // Test dockview panel
      const result2 = await navigationApi.focusPanel('generate', LAUNCHPAD_PANEL_ID);
      expect(result2).toBe(true);
      expect(mockDockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should return false on registration timeout', async () => {
      mockGetAppTab.mockReturnValue('generate');

      // Set a short timeout for testing
      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(false);
    });

    it('should handle errors gracefully', async () => {
      const mockPanel = createMockPanel();

      // Make setActive throw an error
      vi.mocked(mockPanel.api.setActive).mockImplementation(() => {
        throw new Error('Mock error');
      });

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(false);
    });

    it('should work without app connection', async () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      // Don't connect to app
      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });
  });

  describe('Panel Waiting', () => {
    it('should resolve immediately for already registered panels', async () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      const waitPromise = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID);

      await expect(waitPromise).resolves.toBeUndefined();
    });

    it('should handle multiple waiters for same panel', async () => {
      const mockPanel = createMockPanel();

      const waitPromise1 = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID);
      const waitPromise2 = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID);

      setTimeout(() => {
        navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      }, 50);

      await expect(Promise.all([waitPromise1, waitPromise2])).resolves.toEqual([undefined, undefined]);
    });

    it('should timeout if panel is not registered', async () => {
      const waitPromise = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID, 100);

      await expect(waitPromise).rejects.toThrow(/Panel .* registration timed out after 100ms/);
    });

    it('should handle custom timeout', async () => {
      const start = Date.now();
      const waitPromise = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID, 200);

      await expect(waitPromise).rejects.toThrow(/Panel .* registration timed out after 200ms/);

      const elapsed = Date.now() - start;
      // TODO(psyche): Use vitest's fake timeres
      // Allow some margin for timer resolution
      expect(elapsed).toBeGreaterThanOrEqual(190);
      expect(elapsed).toBeLessThan(210);
    });
  });

  describe('Tab Management', () => {
    it('should unregister all panels for a tab', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockPanel();
      const mockPanel3 = createMockPanel();

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel1);
      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockPanel2);
      navigationApi._registerPanel('canvas', SETTINGS_PANEL_ID, mockPanel3);

      expect(navigationApi.getRegisteredPanels('generate')).toHaveLength(2);
      expect(navigationApi.getRegisteredPanels('canvas')).toHaveLength(1);

      navigationApi.unregisterTab('generate');

      expect(navigationApi.getRegisteredPanels('generate')).toHaveLength(0);
      expect(navigationApi.getRegisteredPanels('canvas')).toHaveLength(1);
    });

    it('should clean up pending promises when unregistering tab', async () => {
      const waitPromise = navigationApi.waitForPanel('generate', SETTINGS_PANEL_ID, 5000);

      navigationApi.unregisterTab('generate');

      // The promise should reject with cancellation message since we cleaned up
      await expect(waitPromise).rejects.toThrow('Panel registration cancelled - tab generate was unregistered');
    });
  });

  describe('Integration Tests', () => {
    it('should handle complete workflow', async () => {
      const mockPanel = createMockPanel();

      // Connect to app
      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('canvas');

      // Register panel
      const unregister = navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      // Focus panel (should switch tab and focus)
      const result = await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();

      // Cleanup
      unregister();
      navigationApi.disconnectFromApp();

      expect(navigationApi._app).toBeNull();
      expect(navigationApi.isPanelRegistered('generate', SETTINGS_PANEL_ID)).toBe(false);
    });

    it('should handle multiple panels and tabs', async () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockDockPanel();
      const mockPanel3 = createMockPanel();

      navigationApi.connectToApp(mockAppApi);
      mockGetAppTab.mockReturnValue('generate');

      // Register panels
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel1);
      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockPanel2);
      navigationApi._registerPanel('canvas', WORKSPACE_PANEL_ID, mockPanel3);

      // Focus panels
      await navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);
      expect(mockPanel1.api.setActive).toHaveBeenCalledOnce();

      await navigationApi.focusPanel('generate', LAUNCHPAD_PANEL_ID);
      expect(mockPanel2.api.setActive).toHaveBeenCalledOnce();

      mockGetAppTab.mockReturnValue('generate');
      await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
      expect(mockSetAppTab).toHaveBeenCalledWith('canvas');
      expect(mockPanel3.api.setActive).toHaveBeenCalledOnce();
    });

    it('should handle async registration and focus', async () => {
      const mockPanel = createMockPanel();
      mockGetAppTab.mockReturnValue('generate');

      // Start focusing before registration
      const focusPromise = navigationApi.focusPanel('generate', SETTINGS_PANEL_ID);

      // Register after delay
      setTimeout(() => {
        navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      }, 50);

      const result = await focusPromise;

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });
  });

  describe('focusPanelInActiveTab', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should focus panel in active tab', async () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanelInActiveTab(SETTINGS_PANEL_ID);

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should return false when no active tab', async () => {
      mockGetAppTab.mockReturnValue(null);

      const result = await navigationApi.focusPanelInActiveTab(SETTINGS_PANEL_ID);

      expect(result).toBe(false);
    });

    it('should work without app connection', async () => {
      navigationApi.disconnectFromApp();

      const result = await navigationApi.focusPanelInActiveTab(SETTINGS_PANEL_ID);

      expect(result).toBe(false);
    });
  });

  describe('Panel Expansion and Collapse', () => {
    it('should expand panel with correct constraints and size', () => {
      const mockPanel = createMockPanel();
      const width = 500;

      navigationApi._expandPanel(mockPanel, width);

      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: width,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width });
    });

    it('should collapse panel with zero constraints and size', () => {
      const mockPanel = createMockPanel();

      navigationApi._collapsePanel(mockPanel);

      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: 0,
        minimumWidth: 0,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width: 0 });
    });
  });

  describe('getPanel', () => {
    it('should return registered panel', () => {
      const mockPanel = createMockPanel();
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPanel);

      const result = navigationApi.getPanel('generate', SETTINGS_PANEL_ID);

      expect(result).toBe(mockPanel);
    });

    it('should return undefined for unregistered panel', () => {
      const result = navigationApi.getPanel('generate', 'nonexistent');

      expect(result).toBeUndefined();
    });

    it('should return undefined for non-enabled tab', () => {
      const result = navigationApi.getPanel('models', SETTINGS_PANEL_ID);

      expect(result).toBeUndefined();
    });
  });

  describe('toggleLeftPanel', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should expand collapsed left panel', () => {
      const mockPanel = createMockPanel({ width: 0 });
      navigationApi._registerPanel('generate', LEFT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftPanel();

      expect(result).toBe(true);
      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width: LEFT_PANEL_MIN_SIZE_PX });
    });

    it('should collapse expanded left panel', () => {
      const mockPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      navigationApi._registerPanel('generate', LEFT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftPanel();

      expect(result).toBe(true);
      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: 0,
        minimumWidth: 0,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width: 0 });
    });

    it('should return false when no active tab', () => {
      mockGetAppTab.mockReturnValue(null);

      const result = navigationApi.toggleLeftPanel();

      expect(result).toBe(false);
    });

    it('should return false when left panel not found', () => {
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftPanel();

      expect(result).toBe(false);
    });

    it('should return false when panel is not GridviewPanel', () => {
      const mockPanel = createMockDockPanel();
      navigationApi._registerPanel('generate', LEFT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftPanel();

      expect(result).toBe(false);
    });
  });

  describe('toggleRightPanel', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should expand collapsed right panel', () => {
      const mockPanel = createMockPanel({ width: 0 });
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleRightPanel();

      expect(result).toBe(true);
      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width: RIGHT_PANEL_MIN_SIZE_PX });
    });

    it('should collapse expanded right panel', () => {
      const mockPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleRightPanel();

      expect(result).toBe(true);
      expect(mockPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: 0,
        minimumWidth: 0,
      });
      expect(mockPanel.api.setSize).toHaveBeenCalledWith({ width: 0 });
    });

    it('should return false when no active tab', () => {
      mockGetAppTab.mockReturnValue(null);

      const result = navigationApi.toggleRightPanel();

      expect(result).toBe(false);
    });

    it('should return false when right panel not found', () => {
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleRightPanel();

      expect(result).toBe(false);
    });

    it('should return false when panel is not GridviewPanel', () => {
      const mockPanel = createMockDockPanel();
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleRightPanel();

      expect(result).toBe(false);
    });
  });

  describe('toggleLeftAndRightPanels', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should expand both panels when left is collapsed', () => {
      const leftPanel = createMockPanel({ width: 0 });
      const rightPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(true);

      // Both should be expanded
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
    });

    it('should expand both panels when right is collapsed', () => {
      const leftPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      const rightPanel = createMockPanel({ width: 0 });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(true);

      // Both should be expanded
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
    });

    it('should collapse both panels when both are expanded', () => {
      const leftPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      const rightPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(true);

      // Both should be collapsed
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: 0,
        minimumWidth: 0,
      });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: 0,
        minimumWidth: 0,
      });
    });

    it('should expand both panels when both are collapsed', () => {
      const leftPanel = createMockPanel({ width: 0 });
      const rightPanel = createMockPanel({ width: 0 });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(true);

      // Both should be expanded
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
    });

    it('should return false when no active tab', () => {
      mockGetAppTab.mockReturnValue(null);

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(false);
    });

    it('should return false when panels not found', () => {
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(false);
    });

    it('should return false when panels are not GridviewPanels', () => {
      const leftPanel = createMockDockPanel();
      const rightPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.toggleLeftAndRightPanels();

      expect(result).toBe(false);
    });
  });

  describe('resetLeftAndRightPanels', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should reset both panels to expanded state', () => {
      const leftPanel = createMockPanel({ width: 0 });
      const rightPanel = createMockPanel({ width: 0 });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.resetLeftAndRightPanels();

      expect(result).toBe(true);

      // Both should be reset to expanded state
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(leftPanel.api.setSize).toHaveBeenCalledWith({ width: LEFT_PANEL_MIN_SIZE_PX });

      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
      expect(rightPanel.api.setSize).toHaveBeenCalledWith({ width: RIGHT_PANEL_MIN_SIZE_PX });
    });

    it('should return false when no active tab', () => {
      mockGetAppTab.mockReturnValue(null);

      const result = navigationApi.resetLeftAndRightPanels();

      expect(result).toBe(false);
    });

    it('should return false when panels not found', () => {
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.resetLeftAndRightPanels();

      expect(result).toBe(false);
    });

    it('should return false when panels are not GridviewPanels', () => {
      const leftPanel = createMockDockPanel();
      const rightPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = navigationApi.resetLeftAndRightPanels();

      expect(result).toBe(false);
    });
  });

  describe('Integration Tests', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should handle complete panel management workflow', async () => {
      const leftPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      const rightPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      const settingsPanel = createMockPanel();

      // Register panels
      navigationApi._registerPanel('generate', LEFT_PANEL_ID, leftPanel);
      navigationApi._registerPanel('generate', RIGHT_PANEL_ID, rightPanel);
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, settingsPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Focus a panel in active tab
      const focusResult = await navigationApi.focusPanelInActiveTab(SETTINGS_PANEL_ID);
      expect(focusResult).toBe(true);
      expect(settingsPanel.api.setActive).toHaveBeenCalled();

      // Toggle panels
      navigationApi.toggleLeftAndRightPanels(); // Should collapse both
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({ maximumWidth: 0, minimumWidth: 0 });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({ maximumWidth: 0, minimumWidth: 0 });

      // Reset panels
      navigationApi.resetLeftAndRightPanels(); // Should expand both
      expect(leftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
      expect(rightPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      });
    });

    it('should handle tab switching with panel operations', () => {
      const generateLeftPanel = createMockPanel({ maximumWidth: Number.MAX_SAFE_INTEGER });
      const canvasLeftPanel = createMockPanel({ width: 0 });

      navigationApi._registerPanel('generate', LEFT_PANEL_ID, generateLeftPanel);
      navigationApi._registerPanel('canvas', LEFT_PANEL_ID, canvasLeftPanel);

      // Start on generate tab
      mockGetAppTab.mockReturnValue('generate');
      navigationApi.toggleLeftPanel(); // Should collapse
      expect(generateLeftPanel.api.setConstraints).toHaveBeenCalledWith({ maximumWidth: 0, minimumWidth: 0 });

      // Switch to canvas tab
      mockGetAppTab.mockReturnValue('canvas');
      navigationApi.toggleLeftPanel(); // Should expand
      expect(canvasLeftPanel.api.setConstraints).toHaveBeenCalledWith({
        maximumWidth: Number.MAX_SAFE_INTEGER,
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      });
    });

    it('should handle error cases gracefully', () => {
      mockGetAppTab.mockReturnValue('generate');

      // Test operations without panels registered
      expect(navigationApi.toggleLeftPanel()).toBe(false);
      expect(navigationApi.toggleRightPanel()).toBe(false);
      expect(navigationApi.toggleLeftAndRightPanels()).toBe(false);
      expect(navigationApi.resetLeftAndRightPanels()).toBe(false);
      expect(navigationApi.getPanel('generate', 'nonexistent')).toBeUndefined();
    });

    it('should handle async error cases gracefully', async () => {
      mockGetAppTab.mockReturnValue('generate');

      const focusResult = await navigationApi.focusPanelInActiveTab('nonexistent');
      expect(focusResult).toBe(false);
    });
  });

  describe('registerContainer', () => {
    const tab = 'generate';
    const viewId = 'myView';
    const key = `${tab}:container:${viewId}`;

    beforeEach(() => {
      navigationApi = new NavigationApi();
      navigationApi.connectToApp(mockAppApi);
    });

    it('initializes from scratch when no stored state', () => {
      mockGetStorage.mockReturnValue(undefined);
      const initialize = vi.fn();
      const panel1 = { id: 'p1' };
      const panel2 = { id: 'p2' };
      const mockApi = {
        panels: [panel1, panel2],
        toJSON: vi.fn(() => ({ foo: 'bar' })),
        onDidLayoutChange: vi.fn(() => ({ dispose: vi.fn() })),
      } as unknown as DockviewApi | GridviewApi;
      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      expect(initialize).toHaveBeenCalledOnce();
      expect(mockSetStorage).toHaveBeenCalledOnce();
      expect(mockSetStorage).toHaveBeenCalledWith(key, { foo: 'bar' });
      // panels registered
      expect(navigationApi.isPanelRegistered(tab, 'p1')).toBe(true);
      expect(navigationApi.isPanelRegistered(tab, 'p2')).toBe(true);
    });

    it('restores from storage when fromJSON succeeds', () => {
      const stored = { saved: true };
      mockGetStorage.mockReturnValue(stored);
      const initialize = vi.fn();
      const panel = { id: 'p' };
      const mockApi = {
        panels: [panel],
        fromJSON: vi.fn(),
        toJSON: vi.fn(),
        onDidLayoutChange: vi.fn(() => ({ dispose: vi.fn() })),
      } as unknown as DockviewApi | GridviewApi;
      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      expect(mockApi.fromJSON).toHaveBeenCalledWith(stored);
      expect(initialize).not.toHaveBeenCalled();
      expect(mockSetStorage).not.toHaveBeenCalled(); // no initial persist
      expect(navigationApi.isPanelRegistered(tab, 'p')).toBe(true);
    });

    it('re-initializes when fromJSON throws, deletes then sets', () => {
      const stored = { saved: true };
      mockGetStorage.mockReturnValue(stored);
      const initialize = vi.fn();
      const panel = { id: 'p' };
      const mockApi = {
        panels: [panel],
        fromJSON: vi.fn(() => {
          throw new Error('bad');
        }),
        toJSON: vi.fn(() => ({ new: 'state' })),
        onDidLayoutChange: vi.fn(() => ({ dispose: vi.fn() })),
      } as unknown as DockviewApi | GridviewApi;
      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      expect(mockApi.fromJSON).toHaveBeenCalledWith(stored);
      expect(mockDeleteStorage).toHaveBeenCalledOnce();
      expect(mockDeleteStorage).toHaveBeenCalledWith(key);
      expect(initialize).toHaveBeenCalledOnce();
      expect(mockSetStorage).toHaveBeenCalledOnce();
      expect(mockSetStorage).toHaveBeenCalledWith(key, { new: 'state' });
      expect(navigationApi.isPanelRegistered(tab, 'p')).toBe(true);
    });

    it('persists on layout change after debounce', () => {
      vi.useFakeTimers();
      mockGetStorage.mockReturnValue(undefined);
      const initialize = vi.fn();
      const panel = { id: 'p' };
      let layoutCb: () => void = () => {};
      const mockApi = {
        panels: [panel],
        toJSON: vi.fn(() => ({ x: 1 })),
        onDidLayoutChange: vi.fn((cb) => {
          layoutCb = cb;
          return { dispose: vi.fn() };
        }),
      } as unknown as DockviewApi | GridviewApi;
      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      // first set: initial persistence
      expect(mockSetStorage).toHaveBeenCalledWith(key, { x: 1 });

      // simulate layout change
      layoutCb();
      // advance past debounce (300ms)
      vi.advanceTimersByTime(300);

      expect(mockSetStorage).toHaveBeenCalledTimes(2);
      expect(mockSetStorage).toHaveBeenLastCalledWith(key, { x: 1 });

      vi.useRealTimers();
    });

    it('does nothing if app not connected', () => {
      navigationApi.disconnectFromApp();
      const initialize = vi.fn();
      const mockApi = {
        panels: [],
        fromJSON: vi.fn(),
        toJSON: vi.fn(),
        onDidLayoutChange: vi.fn(),
      } as unknown as DockviewApi | GridviewApi;
      expect(() => navigationApi.registerContainer(tab, viewId, mockApi, initialize)).not.toThrow();
      expect(mockGetStorage).not.toHaveBeenCalled();
      expect(initialize).not.toHaveBeenCalled();
    });
  });
});
