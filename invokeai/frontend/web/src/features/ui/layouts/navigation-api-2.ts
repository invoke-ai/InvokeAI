import { logger } from 'app/logging/logger';
import type { DockviewApi, GridviewApi, IDockviewPanel, IGridviewPanel } from 'dockview';
import type { TabName } from 'features/ui/store/uiTypes';

const log = logger('system');

export type GenerateTabLayout = {
  gridviewApi: Readonly<GridviewApi>;
  panels: {
    left: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        settings: Readonly<IGridviewPanel>;
      };
    };
    main: {
      panelApi: Readonly<IGridviewPanel>;
      dockviewApi: Readonly<DockviewApi>;
      panels: {
        launchpad: Readonly<IDockviewPanel>;
        viewer: Readonly<IDockviewPanel>;
      };
    };
    right: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        boards: Readonly<IGridviewPanel>;
        gallery: Readonly<IGridviewPanel>;
      };
    };
  };
};

export type CanvasTabLayout = {
  gridviewApi: Readonly<GridviewApi>;
  panels: {
    left: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        settings: Readonly<IGridviewPanel>;
      };
    };
    main: {
      panelApi: Readonly<IGridviewPanel>;
      dockviewApi: Readonly<DockviewApi>;
      panels: {
        launchpad: Readonly<IDockviewPanel>;
        workspace: Readonly<IDockviewPanel>;
        viewer: Readonly<IDockviewPanel>;
      };
    };
    right: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        boards: Readonly<IGridviewPanel>;
        gallery: Readonly<IGridviewPanel>;
        layers: Readonly<IGridviewPanel>;
      };
    };
  };
};

export type UpscalingTabLayout = {
  gridviewApi: Readonly<GridviewApi>;
  panels: {
    left: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        settings: Readonly<IGridviewPanel>;
      };
    };
    main: {
      panelApi: Readonly<IGridviewPanel>;
      dockviewApi: Readonly<DockviewApi>;
      panels: {
        launchpad: Readonly<IDockviewPanel>;
        viewer: Readonly<IDockviewPanel>;
      };
    };
    right: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        boards: Readonly<IGridviewPanel>;
        gallery: Readonly<IGridviewPanel>;
      };
    };
  };
};

export type WorkflowsTabLayout = {
  gridviewApi: Readonly<GridviewApi>;
  panels: {
    left: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        settings: Readonly<IGridviewPanel>;
      };
    };
    main: {
      panelApi: Readonly<IGridviewPanel>;
      dockviewApi: Readonly<DockviewApi>;
      panels: {
        launchpad: Readonly<IDockviewPanel>;
        workspace: Readonly<IDockviewPanel>;
        viewer: Readonly<IDockviewPanel>;
      };
    };
    right: {
      panelApi: Readonly<IGridviewPanel>;
      gridviewApi: Readonly<GridviewApi>;
      panels: {
        boards: Readonly<IGridviewPanel>;
        gallery: Readonly<IGridviewPanel>;
      };
    };
  };
};

type AppTabApi = {
  generate: GenerateTabLayout | null;
  canvas: CanvasTabLayout | null;
  upscaling: UpscalingTabLayout | null;
  workflows: WorkflowsTabLayout | null;
};

export class AppNavigationApi {
  appTabApi: AppTabApi = {
    generate: null,
    canvas: null,
    upscaling: null,
    workflows: null,
  };

  setAppTab: ((tab: TabName) => void) | null = null;
  getAppTab: (() => TabName) | null = null;

  private registrationWaiters: Map<keyof AppTabApi, Array<() => void>> = new Map();

  connectToApp(arg: { setAppTab: (tab: TabName) => void; getAppTab: () => TabName }): void {
    const { setAppTab, getAppTab } = arg;
    this.setAppTab = setAppTab;
    this.getAppTab = getAppTab;
  }

  disconnectFromApp(): void {
    this.setAppTab = null;
    this.getAppTab = null;
  }

  registerAppTab<T extends keyof AppTabApi>(tab: T, layout: AppTabApi[T]): () => void {
    this.appTabApi[tab] = layout;

    // Notify any waiting consumers
    const waiters = this.registrationWaiters.get(tab);

    if (waiters) {
      waiters.forEach((resolve) => resolve());
      this.registrationWaiters.delete(tab);
    }

    return () => {
      this.appTabApi[tab] = null;
    };
  }

  /**
   * Focus a specific panel in a specific tab
   * Automatically switches to the target tab if specified
   */
  async focusPanelInTab<
    T extends keyof AppTabApi,
    R extends keyof NonNullable<AppTabApi[T]>['panels'],
    P extends keyof (NonNullable<AppTabApi[T]>['panels'][R] extends { panels: infer Panels } ? Panels : never),
  >(tabName: T, rootPanelId: R, panelId: P): Promise<boolean> {
    try {
      if (this.setAppTab && this.getAppTab && this.getAppTab() !== tabName) {
        this.setAppTab(tabName);
      }

      await this.waitForTabRegistration(tabName);

      const tabLayout = this.appTabApi[tabName];
      if (!tabLayout) {
        log.error(`Tab ${tabName} failed to register`);
        return false;
      }

      const panel = tabLayout.panels[rootPanelId].panels[panelId] as IGridviewPanel | IDockviewPanel;

      panel.api.setActive();

      return true;
    } catch (error) {
      log.error(`Failed to focus panel ${String(panelId)} in tab ${tabName}`);
      return false;
    }
  }

  waitForTabRegistration<T extends keyof AppTabApi>(tabName: T, timeout = 1000): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.appTabApi[tabName]) {
        resolve();
        return;
      }
      let waiters = this.registrationWaiters.get(tabName);
      if (!waiters) {
        waiters = [];
        this.registrationWaiters.set(tabName, waiters);
      }
      waiters.push(resolve);
      const intervalId = setInterval(() => {
        if (this.appTabApi[tabName]) {
          resolve();
        }
      }, 100);
      setTimeout(() => {
        clearInterval(intervalId);
        if (this.appTabApi[tabName]) {
          resolve();
        } else {
          reject(new Error(`Tab ${tabName} registration timed out`));
        }
      }, timeout);
    });
  }
}

export const navigationApi = new AppNavigationApi();
