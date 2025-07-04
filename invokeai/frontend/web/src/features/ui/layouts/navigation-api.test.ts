import type { IDockviewPanel, IGridviewPanel } from 'dockview';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { NavigationApi } from './navigation-api';

// Mock the logger
vi.mock('app/logging/logger', () => ({
  logger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

// Mock panel with setActive method
const createMockPanel = () =>
  ({
    api: {
      setActive: vi.fn(),
    },
  }) as unknown as IGridviewPanel;

const createMockDockPanel = () =>
  ({
    api: {
      setActive: vi.fn(),
    },
  }) as unknown as IDockviewPanel;

describe('AppNavigationApi', () => {
  let navigationApi: NavigationApi;
  let mockSetAppTab: ReturnType<typeof vi.fn>;
  let mockGetAppTab: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    navigationApi = new NavigationApi();
    mockSetAppTab = vi.fn();
    mockGetAppTab = vi.fn();
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
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });

      expect(navigationApi.setAppTab).toBe(mockSetAppTab);
      expect(navigationApi.getAppTab).toBe(mockGetAppTab);
    });

    it('should disconnect from app', () => {
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
      navigationApi.disconnectFromApp();

      expect(navigationApi.setAppTab).toBeNull();
      expect(navigationApi.getAppTab).toBeNull();
    });
  });

  describe('Panel Registration', () => {
    it('should register and unregister panels', () => {
      const mockPanel = createMockPanel();
      const unregister = navigationApi.registerPanel('generate', 'settings', mockPanel);

      expect(typeof unregister).toBe('function');
      expect(navigationApi.isPanelRegistered('generate', 'settings')).toBe(true);

      // Unregister
      unregister();
      expect(navigationApi.isPanelRegistered('generate', 'settings')).toBe(false);
    });

    it('should resolve waiting promises when panel is registered', async () => {
      const mockPanel = createMockPanel();

      // Start waiting for registration
      const waitPromise = navigationApi.waitForPanel('generate', 'settings');

      // Register the panel
      navigationApi.registerPanel('generate', 'settings', mockPanel);

      // Wait should resolve
      await expect(waitPromise).resolves.toBeUndefined();
    });

    it('should handle multiple panels per tab', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockDockPanel();

      navigationApi.registerPanel('generate', 'settings', mockPanel1);
      navigationApi.registerPanel('generate', 'launchpad', mockPanel2);

      expect(navigationApi.isPanelRegistered('generate', 'settings')).toBe(true);
      expect(navigationApi.isPanelRegistered('generate', 'launchpad')).toBe(true);

      const registeredPanels = navigationApi.getRegisteredPanels('generate');
      expect(registeredPanels).toContain('settings');
      expect(registeredPanels).toContain('launchpad');
    });

    it('should handle panels across different tabs', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockPanel();

      navigationApi.registerPanel('generate', 'settings', mockPanel1);
      navigationApi.registerPanel('canvas', 'settings', mockPanel2);

      expect(navigationApi.isPanelRegistered('generate', 'settings')).toBe(true);
      expect(navigationApi.isPanelRegistered('canvas', 'settings')).toBe(true);

      // Same panel ID in different tabs should be separate
      expect(navigationApi.getRegisteredPanels('generate')).toEqual(['settings']);
      expect(navigationApi.getRegisteredPanels('canvas')).toEqual(['settings']);
    });
  });

  describe('Panel Focus', () => {
    beforeEach(() => {
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
    });

    it('should focus panel in already registered tab', async () => {
      const mockPanel = createMockPanel();
      navigationApi.registerPanel('generate', 'settings', mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(true);
      expect(mockSetAppTab).not.toHaveBeenCalled();
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should switch tab before focusing panel', async () => {
      const mockPanel = createMockPanel();
      navigationApi.registerPanel('generate', 'settings', mockPanel);
      mockGetAppTab.mockReturnValue('canvas'); // Currently on different tab

      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should wait for panel registration before focusing', async () => {
      const mockPanel = createMockPanel();
      mockGetAppTab.mockReturnValue('generate');

      // Start focus operation before panel is registered
      const focusPromise = navigationApi.focusPanel('generate', 'settings');

      // Register panel after a short delay
      setTimeout(() => {
        navigationApi.registerPanel('generate', 'settings', mockPanel);
      }, 100);

      const result = await focusPromise;

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should focus different panel types', async () => {
      const mockGridPanel = createMockPanel();
      const mockDockPanel = createMockDockPanel();

      navigationApi.registerPanel('generate', 'settings', mockGridPanel);
      navigationApi.registerPanel('generate', 'launchpad', mockDockPanel);
      mockGetAppTab.mockReturnValue('generate');

      // Test gridview panel
      const result1 = await navigationApi.focusPanel('generate', 'settings');
      expect(result1).toBe(true);
      expect(mockGridPanel.api.setActive).toHaveBeenCalledOnce();

      // Test dockview panel
      const result2 = await navigationApi.focusPanel('generate', 'launchpad');
      expect(result2).toBe(true);
      expect(mockDockPanel.api.setActive).toHaveBeenCalledOnce();
    });

    it('should return false on registration timeout', async () => {
      mockGetAppTab.mockReturnValue('generate');

      // Set a short timeout for testing
      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(false);
    });

    it('should handle errors gracefully', async () => {
      const mockPanel = createMockPanel();

      // Make setActive throw an error
      vi.mocked(mockPanel.api.setActive).mockImplementation(() => {
        throw new Error('Mock error');
      });

      navigationApi.registerPanel('generate', 'settings', mockPanel);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(false);
    });

    it('should work without app connection', async () => {
      const mockPanel = createMockPanel();
      navigationApi.registerPanel('generate', 'settings', mockPanel);

      // Don't connect to app
      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });
  });

  describe('Panel Waiting', () => {
    it('should resolve immediately for already registered panels', async () => {
      const mockPanel = createMockPanel();
      navigationApi.registerPanel('generate', 'settings', mockPanel);

      const waitPromise = navigationApi.waitForPanel('generate', 'settings');

      await expect(waitPromise).resolves.toBeUndefined();
    });

    it('should handle multiple waiters for same panel', async () => {
      const mockPanel = createMockPanel();

      const waitPromise1 = navigationApi.waitForPanel('generate', 'settings');
      const waitPromise2 = navigationApi.waitForPanel('generate', 'settings');

      setTimeout(() => {
        navigationApi.registerPanel('generate', 'settings', mockPanel);
      }, 50);

      await expect(Promise.all([waitPromise1, waitPromise2])).resolves.toEqual([undefined, undefined]);
    });

    it('should timeout if panel is not registered', async () => {
      const waitPromise = navigationApi.waitForPanel('generate', 'settings', 100);

      await expect(waitPromise).rejects.toThrow('Panel generate:settings registration timed out after 100ms');
    });

    it('should handle custom timeout', async () => {
      const start = Date.now();
      const waitPromise = navigationApi.waitForPanel('generate', 'settings', 200);

      await expect(waitPromise).rejects.toThrow('Panel generate:settings registration timed out after 200ms');

      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThanOrEqual(200);
      expect(elapsed).toBeLessThan(300); // Allow some margin
    });
  });

  describe('Tab Management', () => {
    it('should unregister all panels for a tab', () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockPanel();
      const mockPanel3 = createMockPanel();

      navigationApi.registerPanel('generate', 'settings', mockPanel1);
      navigationApi.registerPanel('generate', 'launchpad', mockPanel2);
      navigationApi.registerPanel('canvas', 'settings', mockPanel3);

      expect(navigationApi.getRegisteredPanels('generate')).toHaveLength(2);
      expect(navigationApi.getRegisteredPanels('canvas')).toHaveLength(1);

      navigationApi.unregisterTab('generate');

      expect(navigationApi.getRegisteredPanels('generate')).toHaveLength(0);
      expect(navigationApi.getRegisteredPanels('canvas')).toHaveLength(1);
    });

    it('should clean up pending promises when unregistering tab', async () => {
      const waitPromise = navigationApi.waitForPanel('generate', 'settings', 5000);

      navigationApi.unregisterTab('generate');

      // The promise should reject with cancellation message since we cleaned up
      await expect(waitPromise).rejects.toThrow('Panel registration cancelled - tab generate was unregistered');
    });
  });

  describe('Integration Tests', () => {
    it('should handle complete workflow', async () => {
      const mockPanel = createMockPanel();

      // Connect to app
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
      mockGetAppTab.mockReturnValue('canvas');

      // Register panel
      const unregister = navigationApi.registerPanel('generate', 'settings', mockPanel);

      // Focus panel (should switch tab and focus)
      const result = await navigationApi.focusPanel('generate', 'settings');

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();

      // Cleanup
      unregister();
      navigationApi.disconnectFromApp();

      expect(navigationApi.setAppTab).toBeNull();
      expect(navigationApi.getAppTab).toBeNull();
      expect(navigationApi.isPanelRegistered('generate', 'settings')).toBe(false);
    });

    it('should handle multiple panels and tabs', async () => {
      const mockPanel1 = createMockPanel();
      const mockPanel2 = createMockDockPanel();
      const mockPanel3 = createMockPanel();

      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
      mockGetAppTab.mockReturnValue('generate');

      // Register panels
      navigationApi.registerPanel('generate', 'settings', mockPanel1);
      navigationApi.registerPanel('generate', 'launchpad', mockPanel2);
      navigationApi.registerPanel('canvas', 'workspace', mockPanel3);

      // Focus panels
      await navigationApi.focusPanel('generate', 'settings');
      expect(mockPanel1.api.setActive).toHaveBeenCalledOnce();

      await navigationApi.focusPanel('generate', 'launchpad');
      expect(mockPanel2.api.setActive).toHaveBeenCalledOnce();

      mockGetAppTab.mockReturnValue('generate');
      await navigationApi.focusPanel('canvas', 'workspace');
      expect(mockSetAppTab).toHaveBeenCalledWith('canvas');
      expect(mockPanel3.api.setActive).toHaveBeenCalledOnce();
    });

    it('should handle async registration and focus', async () => {
      const mockPanel = createMockPanel();
      mockGetAppTab.mockReturnValue('generate');

      // Start focusing before registration
      const focusPromise = navigationApi.focusPanel('generate', 'settings');

      // Register after delay
      setTimeout(() => {
        navigationApi.registerPanel('generate', 'settings', mockPanel);
      }, 50);

      const result = await focusPromise;

      expect(result).toBe(true);
      expect(mockPanel.api.setActive).toHaveBeenCalledOnce();
    });
  });
});
