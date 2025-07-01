import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { AutoLayoutPanelContainer } from 'common/components/FocusRegionWrapper';
import type { FocusRegionName } from 'common/hooks/focus';
import type { DockviewApi, GridviewApi, IDockviewPanelProps, IGridviewPanelProps } from 'dockview';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import type { WritableAtom } from 'nanostores';
import { atom } from 'nanostores';
import type { FunctionComponent, PropsWithChildren, RefObject } from 'react';
import { createContext, memo, useCallback, useContext, useMemo, useState } from 'react';

import { LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX } from './shared';

type AutoLayoutContextValue = {
  tab: TabName;
  isActiveTab: boolean;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  toggleBothPanels: () => void;
  resetPanels: () => void;
  focusPanel: (id: string) => void;
  rootRef: RefObject<HTMLDivElement>;
  _$rootPanelApi: WritableAtom<GridviewApi | null>;
  _$leftPanelApi: WritableAtom<GridviewApi | null>;
  _$centerPanelApi: WritableAtom<DockviewApi | null>;
  _$rightPanelApi: WritableAtom<GridviewApi | null>;
};

const AutoLayoutContext = createContext<AutoLayoutContextValue | null>(null);

const expandPanel = (api: GridviewApi, panelId: string, width: number) => {
  const panel = api.getPanel(panelId);
  if (!panel) {
    return;
  }
  panel.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: width });
  panel.api.setSize({ width: width });
};

const collapsePanel = (api: GridviewApi, panelId: string) => {
  const panel = api.getPanel(panelId);
  if (!panel) {
    return;
  }
  panel.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
  panel.api.setSize({ width: 0 });
};

const getIsCollapsed = (api: GridviewApi, panelId: string) => {
  const panel = api.getPanel(panelId);
  if (!panel) {
    return true; // ??
  }
  return panel.maximumWidth === 0;
};

const activatePanel = (api: GridviewApi | DockviewApi, panelId: string) => {
  const panel = api.getPanel(panelId);
  if (!panel) {
    return;
  }
  panel.api.setActive();
};

export const AutoLayoutProvider = (
  props: PropsWithChildren<{
    $rootApi: WritableAtom<GridviewApi | null>;
    rootRef: RefObject<HTMLDivElement>;
    tab: TabName;
  }>
) => {
  const { $rootApi, rootRef, tab, children } = props;
  const selectIsActiveTab = useMemo(() => createSelector(selectActiveTab, (activeTab) => activeTab === tab), [tab]);
  const isActiveTab = useAppSelector(selectIsActiveTab);
  const $leftApi = useState(() => atom<GridviewApi | null>(null))[0];
  const $centerApi = useState(() => atom<DockviewApi | null>(null))[0];
  const $rightApi = useState(() => atom<GridviewApi | null>(null))[0];

  const toggleLeftPanel = useCallback(() => {
    const api = $rootApi.get();
    if (!api) {
      return;
    }
    if (getIsCollapsed(api, LEFT_PANEL_ID)) {
      expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
    } else {
      collapsePanel(api, LEFT_PANEL_ID);
    }
  }, [$rootApi]);

  const toggleRightPanel = useCallback(() => {
    const api = $rootApi.get();
    if (!api) {
      return;
    }
    if (getIsCollapsed(api, RIGHT_PANEL_ID)) {
      expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      collapsePanel(api, RIGHT_PANEL_ID);
    }
  }, [$rootApi]);

  const toggleBothPanels = useCallback(() => {
    const api = $rootApi.get();
    if (!api) {
      return;
    }
    requestAnimationFrame(() => {
      if (getIsCollapsed(api, RIGHT_PANEL_ID) || getIsCollapsed(api, LEFT_PANEL_ID)) {
        expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
        expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
      } else {
        collapsePanel(api, LEFT_PANEL_ID);
        collapsePanel(api, RIGHT_PANEL_ID);
      }
    });
  }, [$rootApi]);

  const resetPanels = useCallback(() => {
    const api = $rootApi.get();
    if (!api) {
      return;
    }
    expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
    expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
  }, [$rootApi]);

  const focusPanel = useCallback(
    (id: string) => {
      const api = $centerApi.get();
      if (!api) {
        return;
      }
      activatePanel(api, id);
    },
    [$centerApi]
  );

  const value = useMemo<AutoLayoutContextValue>(
    () => ({
      tab,
      isActiveTab,
      toggleLeftPanel,
      toggleRightPanel,
      toggleBothPanels,
      resetPanels,
      focusPanel,
      rootRef,
      _$rootPanelApi: $rootApi,
      _$leftPanelApi: $leftApi,
      _$centerPanelApi: $centerApi,
      _$rightPanelApi: $rightApi,
    }),
    [
      tab,
      isActiveTab,
      $centerApi,
      $leftApi,
      $rightApi,
      $rootApi,
      focusPanel,
      resetPanels,
      rootRef,
      toggleBothPanels,
      toggleLeftPanel,
      toggleRightPanel,
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
  const { toggleBothPanels, resetPanels, toggleLeftPanel, toggleRightPanel } = useAutoLayoutContext();
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleLeftPanel',
    callback: toggleLeftPanel,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleRightPanel',
    callback: toggleRightPanel,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'resetPanelLayout',
    callback: resetPanels,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'togglePanels',
    callback: toggleBothPanels,
  });

  return null;
});
PanelHotkeysLogical.displayName = 'PanelHotkeysLogical';

export type PanelParameters = {
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
