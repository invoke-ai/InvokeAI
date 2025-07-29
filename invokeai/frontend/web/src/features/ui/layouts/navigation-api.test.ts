import type { DockviewApi, GridviewApi } from 'dockview';
import { DockviewApi as MockedDockviewApi, DockviewPanel, GridviewPanel } from 'dockview';
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
  VIEWER_PANEL_ID,
  WORKSPACE_PANEL_ID,
} from './shared';

// Mock the logger
vi.mock('app/logging/logger', () => ({
  logger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
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

  // Mock DockviewPanel class for instanceof checks
  class MockDockviewPanel {
    api = {
      setActive: vi.fn(),
      setConstraints: vi.fn(),
      setSize: vi.fn(),
      onDidActiveChange: vi.fn(() => ({ dispose: vi.fn() })),
    };
  }

  // Mock DockviewApi class for instanceof checks
  class MockDockviewApi {
    panels = [];
    activePanel = null;
    toJSON = vi.fn();
    fromJSON = vi.fn();
    onDidLayoutChange = vi.fn();
    onDidActivePanelChange = vi.fn();
  }

  return {
    ...actual,
    GridviewPanel: MockGridviewPanel,
    DockviewPanel: MockDockviewPanel,
    DockviewApi: MockDockviewApi,
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

  describe('toggleViewerPanel', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should switch to viewer panel when not currently on viewer', async () => {
      const mockViewerPanel = createMockDockPanel();
      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current panel to something other than viewer
      navigationApi._currentActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(true);
      expect(mockViewerPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should switch to previous panel when on viewer and previous panel exists', async () => {
      const mockPreviousPanel = createMockDockPanel();
      const mockViewerPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockPreviousPanel);
      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current panel to viewer and previous to settings
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(true);
      expect(mockPreviousPanel.api.setActive).toHaveBeenCalledOnce();
      expect(mockViewerPanel.api.setActive).not.toHaveBeenCalled();
    });

    it('should switch to launchpad when on viewer and no valid previous panel', async () => {
      const mockLaunchpadPanel = createMockDockPanel();
      const mockViewerPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockLaunchpadPanel);
      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current panel to viewer and no previous panel
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', null);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(true);
      expect(mockLaunchpadPanel.api.setActive).toHaveBeenCalledOnce();
      expect(mockViewerPanel.api.setActive).not.toHaveBeenCalled();
    });

    it('should switch to launchpad when on viewer and previous panel is also viewer', async () => {
      const mockLaunchpadPanel = createMockDockPanel();
      const mockViewerPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockLaunchpadPanel);
      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current panel to viewer and previous panel was also viewer
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(true);
      expect(mockLaunchpadPanel.api.setActive).toHaveBeenCalledOnce();
      expect(mockViewerPanel.api.setActive).not.toHaveBeenCalled();
    });

    it('should return false when no active tab', async () => {
      mockGetAppTab.mockReturnValue(null);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(false);
    });

    it('should return false when viewer panel is not registered', async () => {
      mockGetAppTab.mockReturnValue('generate');
      navigationApi._currentActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);

      // Don't register viewer panel
      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(false);
    });

    it('should return false when previous panel is not registered', async () => {
      const mockViewerPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current to viewer and previous to unregistered panel
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', 'unregistered-panel');

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(false);
    });

    it('should return false when launchpad panel is not registered as fallback', async () => {
      const mockViewerPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Set current to viewer and no previous panel, but don't register launchpad
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', null);

      const result = await navigationApi.toggleViewerPanel();

      expect(result).toBe(false);
    });

    it('should work across different tabs independently', async () => {
      const mockViewerPanel1 = createMockDockPanel();
      const mockViewerPanel2 = createMockDockPanel();
      const mockSettingsPanel1 = createMockDockPanel();
      const mockSettingsPanel2 = createMockDockPanel();
      const mockLaunchpadPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel1);
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockSettingsPanel1);
      navigationApi._registerPanel('canvas', VIEWER_PANEL_ID, mockViewerPanel2);
      navigationApi._registerPanel('canvas', SETTINGS_PANEL_ID, mockSettingsPanel2);
      navigationApi._registerPanel('canvas', LAUNCHPAD_PANEL_ID, mockLaunchpadPanel);

      // Set up different states for different tabs
      navigationApi._currentActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);
      navigationApi._currentActiveDockviewPanel.set('canvas', VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('canvas', SETTINGS_PANEL_ID);

      // Test generate tab (should switch to viewer)
      mockGetAppTab.mockReturnValue('generate');
      const result1 = await navigationApi.toggleViewerPanel();
      expect(result1).toBe(true);
      expect(mockViewerPanel1.api.setActive).toHaveBeenCalledOnce();

      // Test canvas tab (should switch to previous panel - settings panel in canvas)
      mockGetAppTab.mockReturnValue('canvas');
      const result2 = await navigationApi.toggleViewerPanel();
      expect(result2).toBe(true);
      expect(mockSettingsPanel2.api.setActive).toHaveBeenCalledOnce();
    });

    it('should handle sequence of viewer toggles correctly', async () => {
      const mockViewerPanel = createMockDockPanel();
      const mockSettingsPanel = createMockDockPanel();
      const mockLaunchpadPanel = createMockDockPanel();

      navigationApi._registerPanel('generate', VIEWER_PANEL_ID, mockViewerPanel);
      navigationApi._registerPanel('generate', SETTINGS_PANEL_ID, mockSettingsPanel);
      navigationApi._registerPanel('generate', LAUNCHPAD_PANEL_ID, mockLaunchpadPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Start on settings panel
      navigationApi._currentActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set('generate', null);

      // First toggle: settings -> viewer
      const result1 = await navigationApi.toggleViewerPanel();
      expect(result1).toBe(true);
      expect(mockViewerPanel.api.setActive).toHaveBeenCalledOnce();

      // Simulate panel change tracking (normally done by dockview listener)
      navigationApi._prevActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);
      navigationApi._currentActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);

      // Second toggle: viewer -> settings (previous panel)
      const result2 = await navigationApi.toggleViewerPanel();
      expect(result2).toBe(true);
      expect(mockSettingsPanel.api.setActive).toHaveBeenCalledOnce();

      // Simulate panel change tracking again
      navigationApi._prevActiveDockviewPanel.set('generate', VIEWER_PANEL_ID);
      navigationApi._currentActiveDockviewPanel.set('generate', SETTINGS_PANEL_ID);

      // Third toggle: settings -> viewer again
      const result3 = await navigationApi.toggleViewerPanel();
      expect(result3).toBe(true);
      expect(mockViewerPanel.api.setActive).toHaveBeenCalledTimes(2);
    });
  });

  describe('Disposable Cleanup', () => {
    beforeEach(() => {
      navigationApi.connectToApp(mockAppApi);
    });

    it('should add disposable functions for a tab', () => {
      const dispose1 = vi.fn();
      const dispose2 = vi.fn();

      navigationApi._addDisposeForTab('generate', dispose1);
      navigationApi._addDisposeForTab('generate', dispose2);

      // Check that disposables are stored
      const disposables = navigationApi._disposablesForTab.get('generate');
      expect(disposables).toBeDefined();
      expect(disposables?.size).toBe(2);
      expect(disposables?.has(dispose1)).toBe(true);
      expect(disposables?.has(dispose2)).toBe(true);
    });

    it('should handle multiple tabs independently', () => {
      const dispose1 = vi.fn();
      const dispose2 = vi.fn();
      const dispose3 = vi.fn();

      navigationApi._addDisposeForTab('generate', dispose1);
      navigationApi._addDisposeForTab('generate', dispose2);
      navigationApi._addDisposeForTab('canvas', dispose3);

      const generateDisposables = navigationApi._disposablesForTab.get('generate');
      const canvasDisposables = navigationApi._disposablesForTab.get('canvas');

      expect(generateDisposables?.size).toBe(2);
      expect(canvasDisposables?.size).toBe(1);
      expect(generateDisposables?.has(dispose1)).toBe(true);
      expect(generateDisposables?.has(dispose2)).toBe(true);
      expect(canvasDisposables?.has(dispose3)).toBe(true);
    });

    it('should call all dispose functions when unregistering a tab', () => {
      const dispose1 = vi.fn();
      const dispose2 = vi.fn();
      const dispose3 = vi.fn();

      // Add disposables for generate tab
      navigationApi._addDisposeForTab('generate', dispose1);
      navigationApi._addDisposeForTab('generate', dispose2);

      // Add disposable for canvas tab (should not be called)
      navigationApi._addDisposeForTab('canvas', dispose3);

      // Unregister generate tab
      navigationApi.unregisterTab('generate');

      // Check that generate tab disposables were called
      expect(dispose1).toHaveBeenCalledOnce();
      expect(dispose2).toHaveBeenCalledOnce();

      // Check that canvas tab disposable was not called
      expect(dispose3).not.toHaveBeenCalled();

      // Check that generate tab disposables are cleared
      expect(navigationApi._disposablesForTab.has('generate')).toBe(false);

      // Check that canvas tab disposables remain
      expect(navigationApi._disposablesForTab.has('canvas')).toBe(true);
    });

    it('should handle unregistering tab with no disposables gracefully', () => {
      // Should not throw when unregistering tab with no disposables
      expect(() => navigationApi.unregisterTab('generate')).not.toThrow();
    });

    it('should handle duplicate dispose functions', () => {
      const dispose1 = vi.fn();

      // Add the same dispose function twice
      navigationApi._addDisposeForTab('generate', dispose1);
      navigationApi._addDisposeForTab('generate', dispose1);

      const disposables = navigationApi._disposablesForTab.get('generate');
      // Set should contain only one instance (sets don't allow duplicates)
      expect(disposables?.size).toBe(1);

      navigationApi.unregisterTab('generate');

      // Should be called only once despite being added twice
      expect(dispose1).toHaveBeenCalledOnce();
    });

    it('should automatically add dispose functions during container registration with DockviewApi', () => {
      const tab = 'generate';
      const viewId = 'myView';
      mockGetStorage.mockReturnValue(undefined);

      const initialize = vi.fn();
      const panel = { id: 'p1' };
      const mockDispose = vi.fn();

      // Create a mock that will pass the instanceof DockviewApi check
      const mockApi = Object.create(MockedDockviewApi.prototype);
      Object.assign(mockApi, {
        panels: [panel],
        activePanel: { id: 'p1' },
        toJSON: vi.fn(() => ({ foo: 'bar' })),
        onDidLayoutChange: vi.fn(() => ({ dispose: vi.fn() })),
        onDidActivePanelChange: vi.fn(() => ({ dispose: mockDispose })),
      });

      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      // Check that dispose function was added to disposables
      const disposables = navigationApi._disposablesForTab.get(tab);
      expect(disposables).toBeDefined();
      expect(disposables?.size).toBe(1);

      // Unregister tab and check dispose was called
      navigationApi.unregisterTab(tab);
      expect(mockDispose).toHaveBeenCalledOnce();
    });

    it('should not add dispose functions for GridviewApi during container registration', () => {
      const tab = 'generate';
      const viewId = 'myView';
      mockGetStorage.mockReturnValue(undefined);

      const initialize = vi.fn();
      const panel = { id: 'p1' };

      // Mock GridviewApi (not DockviewApi)
      const mockApi = {
        panels: [panel],
        toJSON: vi.fn(() => ({ foo: 'bar' })),
        onDidLayoutChange: vi.fn(() => ({ dispose: vi.fn() })),
      } as unknown as GridviewApi;

      navigationApi.registerContainer(tab, viewId, mockApi, initialize);

      // Check that no dispose function was added for GridviewApi
      const disposables = navigationApi._disposablesForTab.get(tab);
      expect(disposables).toBeUndefined();
    });

    it('should handle dispose function errors gracefully', () => {
      const goodDispose = vi.fn();
      const errorDispose = vi.fn(() => {
        throw new Error('Dispose error');
      });
      const anotherGoodDispose = vi.fn();

      navigationApi._addDisposeForTab('generate', goodDispose);
      navigationApi._addDisposeForTab('generate', errorDispose);
      navigationApi._addDisposeForTab('generate', anotherGoodDispose);

      // Should not throw even if one dispose function throws
      expect(() => navigationApi.unregisterTab('generate')).not.toThrow();

      // All dispose functions should have been called
      expect(goodDispose).toHaveBeenCalledOnce();
      expect(errorDispose).toHaveBeenCalledOnce();
      expect(anotherGoodDispose).toHaveBeenCalledOnce();
    });

    it('should clear panel tracking state when unregistering tab', () => {
      const tab = 'generate';

      // Set up some panel tracking state
      navigationApi._currentActiveDockviewPanel.set(tab, VIEWER_PANEL_ID);
      navigationApi._prevActiveDockviewPanel.set(tab, SETTINGS_PANEL_ID);

      // Add some disposables
      const dispose1 = vi.fn();
      const dispose2 = vi.fn();
      navigationApi._addDisposeForTab(tab, dispose1);
      navigationApi._addDisposeForTab(tab, dispose2);

      // Verify state exists before unregistering
      expect(navigationApi._currentActiveDockviewPanel.has(tab)).toBe(true);
      expect(navigationApi._prevActiveDockviewPanel.has(tab)).toBe(true);
      expect(navigationApi._disposablesForTab.has(tab)).toBe(true);

      // Unregister tab
      navigationApi.unregisterTab(tab);

      // Verify all state is cleared
      expect(navigationApi._currentActiveDockviewPanel.has(tab)).toBe(false);
      expect(navigationApi._prevActiveDockviewPanel.has(tab)).toBe(false);
      expect(navigationApi._disposablesForTab.has(tab)).toBe(false);

      // Verify dispose functions were called
      expect(dispose1).toHaveBeenCalledOnce();
      expect(dispose2).toHaveBeenCalledOnce();
    });
  });
});
