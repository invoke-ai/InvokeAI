import type { DockviewApi, GridviewApi, IDockviewPanel, IGridviewPanel } from 'dockview';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GenerateTabLayout } from './navigation-api-2';
import { AppNavigationApi } from './navigation-api-2';

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
    // Add other required properties as needed
  }) as unknown as IGridviewPanel;

const createMockDockPanel = () =>
  ({
    api: {
      setActive: vi.fn(),
    },
    // Add other required properties as needed
  }) as unknown as IDockviewPanel;

// Create a mock layout for testing
const createMockGenerateLayout = (): GenerateTabLayout => ({
  gridviewApi: {} as Readonly<GridviewApi>,
  panels: {
    left: {
      panelApi: {} as Readonly<IGridviewPanel>,
      gridviewApi: {} as Readonly<GridviewApi>,
      panels: {
        settings: createMockPanel(),
      },
    },
    main: {
      panelApi: {} as Readonly<IGridviewPanel>,
      dockviewApi: {} as Readonly<DockviewApi>,
      panels: {
        launchpad: createMockDockPanel(),
        viewer: createMockDockPanel(),
      },
    },
    right: {
      panelApi: {} as Readonly<IGridviewPanel>,
      gridviewApi: {} as Readonly<GridviewApi>,
      panels: {
        boards: createMockPanel(),
        gallery: createMockPanel(),
      },
    },
  },
});

describe('AppNavigationApi', () => {
  let navigationApi: AppNavigationApi;
  let mockSetAppTab: ReturnType<typeof vi.fn>;
  let mockGetAppTab: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    navigationApi = new AppNavigationApi();
    mockSetAppTab = vi.fn();
    mockGetAppTab = vi.fn();
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

  describe('Tab Registration', () => {
    it('should register and unregister tabs', () => {
      const mockLayout = createMockGenerateLayout();
      const unregister = navigationApi.registerAppTab('generate', mockLayout);

      expect(typeof unregister).toBe('function');

      // Check that tab is registered using type assertion to access private property
      expect(navigationApi.appTabApi.generate).toBe(mockLayout);

      // Unregister
      unregister();
      expect(navigationApi.appTabApi.generate).toBeNull();
    });

    it('should notify waiters when tab is registered', async () => {
      const mockLayout = createMockGenerateLayout();

      // Start waiting for registration
      const waitPromise = navigationApi.waitForTabRegistration('generate');

      // Register the tab
      navigationApi.registerAppTab('generate', mockLayout);

      // Wait should resolve
      await expect(waitPromise).resolves.toBeUndefined();
    });
  });

  describe('Panel Focus', () => {
    beforeEach(() => {
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
    });

    it('should focus panel in already registered tab', async () => {
      const mockLayout = createMockGenerateLayout();
      navigationApi.registerAppTab('generate', mockLayout);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanelInTab('generate', 'left', 'settings');

      expect(result).toBe(true);
      expect(mockSetAppTab).not.toHaveBeenCalled();
      expect(mockLayout.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();
    });

    it('should switch tab before focusing panel', async () => {
      const mockLayout = createMockGenerateLayout();
      navigationApi.registerAppTab('generate', mockLayout);
      mockGetAppTab.mockReturnValue('canvas'); // Currently on different tab

      const result = await navigationApi.focusPanelInTab('generate', 'left', 'settings');

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockLayout.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();
    });

    it('should wait for tab registration before focusing', async () => {
      const mockLayout = createMockGenerateLayout();
      mockGetAppTab.mockReturnValue('generate');

      // Start focus operation before tab is registered
      const focusPromise = navigationApi.focusPanelInTab('generate', 'left', 'settings');

      // Register tab after a short delay
      setTimeout(() => {
        navigationApi.registerAppTab('generate', mockLayout);
      }, 100);

      const result = await focusPromise;

      expect(result).toBe(true);
      expect(mockLayout.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();
    });

    it('should focus different panel types', async () => {
      const mockLayout = createMockGenerateLayout();
      navigationApi.registerAppTab('generate', mockLayout);
      mockGetAppTab.mockReturnValue('generate');

      // Test gridview panel
      const result1 = await navigationApi.focusPanelInTab('generate', 'left', 'settings');
      expect(result1).toBe(true);
      expect(mockLayout.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();

      // Test dockview panel
      const result2 = await navigationApi.focusPanelInTab('generate', 'main', 'launchpad');
      expect(result2).toBe(true);
      expect(mockLayout.panels.main.panels.launchpad.api.setActive).toHaveBeenCalledOnce();

      // Test right panel
      const result3 = await navigationApi.focusPanelInTab('generate', 'right', 'boards');
      expect(result3).toBe(true);
      expect(mockLayout.panels.right.panels.boards.api.setActive).toHaveBeenCalledOnce();
    });

    it('should return false on registration timeout', async () => {
      mockGetAppTab.mockReturnValue('generate');

      // Set a short timeout for testing
      const result = await navigationApi.focusPanelInTab('generate', 'left', 'settings');

      expect(result).toBe(false);
    });

    it('should handle errors gracefully', async () => {
      const mockLayout = createMockGenerateLayout();

      // Make setActive throw an error
      vi.mocked(mockLayout.panels.left.panels.settings.api.setActive).mockImplementation(() => {
        throw new Error('Mock error');
      });

      navigationApi.registerAppTab('generate', mockLayout);
      mockGetAppTab.mockReturnValue('generate');

      const result = await navigationApi.focusPanelInTab('generate', 'left', 'settings');

      expect(result).toBe(false);
    });

    it('should work without app connection', async () => {
      const mockLayout = createMockGenerateLayout();
      navigationApi.registerAppTab('generate', mockLayout);

      // Don't connect to app
      const result = await navigationApi.focusPanelInTab('generate', 'left', 'settings');

      expect(result).toBe(true);
      expect(mockLayout.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();
    });
  });

  describe('Registration Waiting', () => {
    it('should resolve immediately for already registered tabs', async () => {
      const mockLayout = createMockGenerateLayout();
      navigationApi.registerAppTab('generate', mockLayout);

      const waitPromise = navigationApi.waitForTabRegistration('generate');

      await expect(waitPromise).resolves.toBeUndefined();
    });

    it('should handle multiple waiters', async () => {
      const mockLayout = createMockGenerateLayout();

      const waitPromise1 = navigationApi.waitForTabRegistration('generate');
      const waitPromise2 = navigationApi.waitForTabRegistration('generate');

      setTimeout(() => {
        navigationApi.registerAppTab('generate', mockLayout);
      }, 50);

      await expect(Promise.all([waitPromise1, waitPromise2])).resolves.toEqual([undefined, undefined]);
    });

    it('should timeout if tab is not registered', async () => {
      const waitPromise = navigationApi.waitForTabRegistration('generate', 100);

      await expect(waitPromise).rejects.toThrow('Tab generate registration timed out');
    });
  });

  describe('Integration Tests', () => {
    it('should handle complete workflow', async () => {
      const mockLayout = createMockGenerateLayout();

      // Connect to app
      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
      mockGetAppTab.mockReturnValue('canvas');

      // Register tab
      const unregister = navigationApi.registerAppTab('generate', mockLayout);

      // Focus panel (should switch tab and focus)
      const result = await navigationApi.focusPanelInTab('generate', 'right', 'gallery');

      expect(result).toBe(true);
      expect(mockSetAppTab).toHaveBeenCalledWith('generate');
      expect(mockLayout.panels.right.panels.gallery.api.setActive).toHaveBeenCalledOnce();

      // Cleanup
      unregister();
      navigationApi.disconnectFromApp();

      expect(navigationApi.setAppTab).toBeNull();
      expect(navigationApi.getAppTab).toBeNull();
    });

    it('should handle multiple tabs sequentially', async () => {
      const mockLayout1 = createMockGenerateLayout();
      const mockLayout2 = createMockGenerateLayout();

      navigationApi.connectToApp({ setAppTab: mockSetAppTab, getAppTab: mockGetAppTab });
      mockGetAppTab.mockReturnValue('generate');

      // Register first tab
      const unregister1 = navigationApi.registerAppTab('generate', mockLayout1);

      // Focus panel in first tab
      await navigationApi.focusPanelInTab('generate', 'left', 'settings');
      expect(mockLayout1.panels.left.panels.settings.api.setActive).toHaveBeenCalledOnce();

      // Replace with second tab
      unregister1();
      navigationApi.registerAppTab('generate', mockLayout2);

      // Focus panel in second tab
      await navigationApi.focusPanelInTab('generate', 'right', 'boards');
      expect(mockLayout2.panels.right.panels.boards.api.setActive).toHaveBeenCalledOnce();
    });
  });
});
