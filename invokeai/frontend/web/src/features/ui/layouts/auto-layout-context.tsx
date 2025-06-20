import type { GridviewApi } from 'dockview';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { Atom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useMemo } from 'react';

import { LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX } from './shared';

type AutoLayoutContextValue = {
  $api: Atom<GridviewApi | null>;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  toggleBothPanels: () => void;
  resetPanels: () => void;
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

export const AutoLayoutProvider = (props: PropsWithChildren<{ $api: Atom<GridviewApi | null> }>) => {
  const toggleLeftPanel = useCallback(() => {
    const api = props.$api.get();
    if (!api) {
      return;
    }
    if (getIsCollapsed(api, LEFT_PANEL_ID)) {
      expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
    } else {
      collapsePanel(api, LEFT_PANEL_ID);
    }
  }, [props.$api]);

  const toggleRightPanel = useCallback(() => {
    const api = props.$api.get();
    if (!api) {
      return;
    }
    if (getIsCollapsed(api, RIGHT_PANEL_ID)) {
      expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
    } else {
      collapsePanel(api, RIGHT_PANEL_ID);
    }
  }, [props.$api]);

  const toggleBothPanels = useCallback(() => {
    const api = props.$api.get();
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
  }, [props.$api]);

  const resetPanels = useCallback(() => {
    const api = props.$api.get();
    if (!api) {
      return;
    }
    expandPanel(api, LEFT_PANEL_ID, LEFT_PANEL_MIN_SIZE_PX);
    expandPanel(api, RIGHT_PANEL_ID, RIGHT_PANEL_MIN_SIZE_PX);
  }, [props.$api]);

  const value = useMemo<AutoLayoutContextValue>(
    () => ({
      $api: props.$api,
      toggleLeftPanel,
      toggleRightPanel,
      toggleBothPanels,
      resetPanels,
    }),
    [props.$api, resetPanels, toggleBothPanels, toggleLeftPanel, toggleRightPanel]
  );
  return <AutoLayoutContext.Provider value={value}>{props.children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const value = useContext(AutoLayoutContext);
  if (!value) {
    throw new Error('useAutoLayoutContext must be used within an AutoLayoutProvider');
  }
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
