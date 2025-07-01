import type { FocusRegionName } from 'common/hooks/focus';
import type { IDockviewPanelProps, IGridviewPanelProps } from 'dockview';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { TabName } from 'features/ui/store/uiTypes';
import type { FunctionComponent, PropsWithChildren, RefObject } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';

import { AutoLayoutPanelContainer } from './AutoLayoutPanelContainer';
import { panelRegistry } from './panel-registry/panelApiRegistry';

type AutoLayoutContextValue = {
  tab: TabName;
  // isActiveTab: boolean;
  // toggleLeftPanel: () => void;
  // toggleRightPanel: () => void;
  // toggleBothPanels: () => void;
  // resetPanels: () => void;
  // focusPanel: (id: string) => void;
  rootRef: RefObject<HTMLDivElement>;
  // _$rootPanelApi: WritableAtom<GridviewApi | null>;
  // _$leftPanelApi: WritableAtom<GridviewApi | null>;
  // _$centerPanelApi: WritableAtom<DockviewApi | null>;
  // _$rightPanelApi: WritableAtom<GridviewApi | null>;
  // // Global registry access for cross-tab operations
  // registry: typeof panelApiRegistry;
};

const AutoLayoutContext = createContext<AutoLayoutContextValue | null>(null);

// const expandPanel = (api: GridviewApi, panelId: string, width: number) => {
//   const panel = api.getPanel(panelId);
//   if (!panel) {
//     return;
//   }
//   panel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: width });
//   panel.api.setSize({ width: width });
// };

// const collapsePanel = (api: GridviewApi, panelId: string) => {
//   const panel = api.getPanel(panelId);
//   if (!panel) {
//     return;
//   }
//   panel.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
//   panel.api.setSize({ width: 0 });
// };

// const getIsCollapsed = (api: GridviewApi, panelId: string) => {
//   const panel = api.getPanel(panelId);
//   if (!panel) {
//     return true; // ??
//   }
//   return panel.maximumWidth === 0;
// };

// const activatePanel = (api: GridviewApi | DockviewApi, panelId: string) => {
//   const panel = api.getPanel(panelId);
//   if (!panel) {
//     return;
//   }
//   panel.api.setActive();
// };

export const AutoLayoutProvider = (
  props: PropsWithChildren<{
    // $rootApi: WritableAtom<GridviewApi | null>;
    rootRef: RefObject<HTMLDivElement>;
    tab: TabName;
  }>
) => {
  const { tab, rootRef, children } = props;
  // const { $rootApi, rootRef, tab, children } = props;
  // const selectIsActiveTab = useMemo(() => createSelector(selectActiveTab, (activeTab) => activeTab === tab), [tab]);
  // const isActiveTab = useAppSelector(selectIsActiveTab);
  // const $leftApi = useState(() => atom<GridviewApi | null>(null))[0];
  // const $centerApi = useState(() => atom<DockviewApi | null>(null))[0];
  // const $rightApi = useState(() => atom<GridviewApi | null>(null))[0];

  // // Register this tab with the global panel registry when APIs are available
  // useEffect(() => {
  //   const rootApi = $rootApi.get();
  //   const leftApi = $leftApi.get();
  //   const centerApi = $centerApi.get();
  //   const rightApi = $rightApi.get();

  //   if (rootApi) {
  //     panelApiRegistry.registerTab(tab, {
  //       root: rootApi,
  //       left: leftApi,
  //       center: centerApi,
  //       right: rightApi,
  //     });

  //     return () => {
  //       panelApiRegistry.unregisterTab(tab);
  //     };
  //   }
  // }, [tab, $rootApi, $leftApi, $centerApi, $rightApi]);

  // // Subscribe to API changes and update registry
  // useEffect(() => {
  //   const unsubscribeRoot = $rootApi.subscribe((rootApi) => {
  //     if (rootApi) {
  //       panelApiRegistry.registerTab(tab, {
  //         root: rootApi,
  //         left: $leftApi.get(),
  //         center: $centerApi.get(),
  //         right: $rightApi.get(),
  //       });
  //     }
  //   });

  //   const unsubscribeLeft = $leftApi.subscribe((leftApi) => {
  //     const rootApi = $rootApi.get();
  //     if (rootApi) {
  //       panelApiRegistry.registerTab(tab, {
  //         root: rootApi,
  //         left: leftApi,
  //         center: $centerApi.get(),
  //         right: $rightApi.get(),
  //       });
  //     }
  //   });

  //   const unsubscribeCenter = $centerApi.subscribe((centerApi) => {
  //     const rootApi = $rootApi.get();
  //     if (rootApi) {
  //       panelApiRegistry.registerTab(tab, {
  //         root: rootApi,
  //         left: $leftApi.get(),
  //         center: centerApi,
  //         right: $rightApi.get(),
  //       });
  //     }
  //   });

  //   const unsubscribeRight = $rightApi.subscribe((rightApi) => {
  //     const rootApi = $rootApi.get();
  //     if (rootApi) {
  //       panelApiRegistry.registerTab(tab, {
  //         root: rootApi,
  //         left: $leftApi.get(),
  //         center: $centerApi.get(),
  //         right: rightApi,
  //       });
  //     }
  //   });

  //   return () => {
  //     unsubscribeRoot();
  //     unsubscribeLeft();
  //     unsubscribeCenter();
  //     unsubscribeRight();
  //   };
  // }, [tab, $rootApi, $leftApi, $centerApi, $rightApi]);

  // const toggleLeftPanel = useCallback(() => {
  //   const api = $rootApi.get();
  //   if (!api) {
  //     return;
  //   }
  //   if (getIsCollapsed(api, LEFT_PANEL_ID)) {
  //     expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
  //   } else {
  //     collapsePanel(api, LEFT_PANEL_ID);
  //   }
  // }, [$rootApi]);

  // const toggleRightPanel = useCallback(() => {
  //   const api = $rootApi.get();
  //   if (!api) {
  //     return;
  //   }
  //   if (getIsCollapsed(api, RIGHT_PANEL_ID)) {
  //     expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
  //   } else {
  //     collapsePanel(api, RIGHT_PANEL_ID);
  //   }
  // }, [$rootApi]);

  // const toggleBothPanels = useCallback(() => {
  //   const api = $rootApi.get();
  //   if (!api) {
  //     return;
  //   }
  //   requestAnimationFrame(() => {
  //     if (getIsCollapsed(api, RIGHT_PANEL_ID) || getIsCollapsed(api, LEFT_PANEL_ID)) {
  //       expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
  //       expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
  //     } else {
  //       collapsePanel(api, LEFT_PANEL_ID);
  //       collapsePanel(api, RIGHT_PANEL_ID);
  //     }
  //   });
  // }, [$rootApi]);

  // const resetPanels = useCallback(() => {
  //   const api = $rootApi.get();
  //   if (!api) {
  //     return;
  //   }
  //   expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
  //   expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
  // }, [$rootApi]);

  // const focusPanel = useCallback(
  //   (id: string) => {
  //     const api = $centerApi.get();
  //     if (!api) {
  //       return;
  //     }
  //     activatePanel(api, id);
  //   },
  //   [$centerApi]
  // );

  const value = useMemo<AutoLayoutContextValue>(
    () => ({
      tab,
      // isActiveTab,
      // toggleLeftPanel,
      // toggleRightPanel,
      // toggleBothPanels,
      // resetPanels,
      // focusPanel,
      rootRef,
      // _$rootPanelApi: $rootApi,
      // _$leftPanelApi: $leftApi,
      // _$centerPanelApi: $centerApi,
      // _$rightPanelApi: $rightApi,
      // registry: panelApiRegistry,
    }),
    [
      tab,
      // isActiveTab,
      // $centerApi,
      // $leftApi,
      // $rightApi,
      // $rootApi,
      // focusPanel,
      // resetPanels,
      rootRef,
      // toggleBothPanels,
      // toggleLeftPanel,
      // toggleRightPanel,
    ]
  );
  return <AutoLayoutContext.Provider value={value}>{children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const value = useContext(AutoLayoutContext);
  if (!value) {
    throw new Error('useAutoLayoutContext must be used within an AutoLayoutProvider');
  }
  return value;
};

export const useAutoLayoutContextSafe = () => {
  const value = useContext(AutoLayoutContext);
  return value;
};

export const PanelHotkeysLogical = memo(() => {
  const { tab } = useAutoLayoutContext();

  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleLeftPanel',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleLeftPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleRightPanel',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleRightPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'resetPanelLayout',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.resetPanelsInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'togglePanels',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleBothPanelsInTab(tab);
    },
    dependencies: [tab],
  });

  return null;
});
PanelHotkeysLogical.displayName = 'PanelHotkeysLogical';

export type PanelParameters = {
  tab: TabName;
  focusRegion: FocusRegionName;
};

export type AutoLayoutGridviewComponents = Record<string, FunctionComponent<IGridviewPanelProps<PanelParameters>>>;
export type AutoLayoutDockviewComponents = Record<string, FunctionComponent<IDockviewPanelProps<PanelParameters>>>;
export type RootLayoutGridviewComponents = Record<string, FunctionComponent<IGridviewPanelProps<PanelParameters>>>;
export type PanelProps = IDockviewPanelProps<PanelParameters> | IGridviewPanelProps<PanelParameters>;

export const withPanelContainer = (Component: FunctionComponent) =>
  memo((props: PanelProps) => {
    return (
      <AutoLayoutPanelContainer {...props}>
        <Component />
      </AutoLayoutPanelContainer>
    );
  });
