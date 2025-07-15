import { GridviewPanel, type GridviewPanelApi, type IGridviewPanel } from 'dockview';
import type { TabName } from 'features/ui/store/uiTypes';
import { atom } from 'nanostores';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { navigationApi } from './navigation-api';

const getIsCollapsed = (
  panel: IGridviewPanel<GridviewPanelApi>,
  orientation: 'vertical' | 'horizontal',
  collapsedSize?: number
) => {
  if (orientation === 'vertical') {
    return panel.height <= (collapsedSize ?? panel.minimumHeight ?? 0);
  }
  return panel.width <= (collapsedSize ?? panel.minimumWidth ?? 0);
};

export const useCollapsibleGridviewPanel = (
  tab: TabName,
  panelId: string,
  orientation: 'horizontal' | 'vertical',
  defaultSize: number,
  collapsedSize?: number,
  minExpandedSize?: number
) => {
  const $isCollapsed = useState(() => atom(false))[0];
  const lastExpandedSizeRef = useRef<number>(0);
  const collapse = useCallback(() => {
    const panel = navigationApi.getPanel(tab, panelId);

    if (!panel || !(panel instanceof GridviewPanel)) {
      return;
    }

    lastExpandedSizeRef.current = orientation === 'vertical' ? panel.height : panel.width;

    if (orientation === 'vertical') {
      panel.api.setSize({ height: collapsedSize ?? panel.minimumHeight ?? 0 });
    } else {
      panel.api.setSize({ width: collapsedSize ?? panel.minimumWidth ?? 0 });
    }
  }, [collapsedSize, orientation, panelId, tab]);

  const expand = useCallback(() => {
    const panel = navigationApi.getPanel(tab, panelId);
    if (!panel || !(panel instanceof GridviewPanel)) {
      return;
    }

    let newSize = lastExpandedSizeRef.current || defaultSize;
    if (minExpandedSize && newSize < minExpandedSize) {
      newSize = minExpandedSize;
    }

    if (orientation === 'vertical') {
      panel.api.setSize({ height: newSize });
    } else {
      panel.api.setSize({ width: newSize });
    }
  }, [defaultSize, minExpandedSize, orientation, panelId, tab]);

  const toggle = useCallback(() => {
    const panel = navigationApi.getPanel(tab, panelId);
    if (!panel || !(panel instanceof GridviewPanel)) {
      return;
    }
    const isCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
    if (isCollapsed) {
      expand();
    } else {
      collapse();
    }
  }, [tab, panelId, orientation, collapsedSize, expand, collapse]);

  useEffect(() => {
    const panel = navigationApi.getPanel(tab, panelId);
    if (!panel || !(panel instanceof GridviewPanel)) {
      return;
    }

    const disposable = panel.api.onDidDimensionsChange(() => {
      const isCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
      $isCollapsed.set(isCollapsed);
    });

    return () => {
      disposable.dispose();
    };
  }, [$isCollapsed, collapsedSize, orientation, panelId, tab]);

  return useMemo(
    () => ({
      $isCollapsed,
      expand,
      collapse,
      toggle,
    }),
    [$isCollapsed, collapse, expand, toggle]
  );
};
