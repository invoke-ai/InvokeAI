import type { GridviewPanelApi, IGridviewPanel } from 'dockview';
import type { TabName } from 'features/ui/store/uiTypes';
import { atom } from 'nanostores';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { TabPanelApis } from './panel-registry/panelApiRegistry';
import { panelRegistry } from './panel-registry/panelApiRegistry';

const getIsCollapsed = (
  panel: IGridviewPanel<GridviewPanelApi>,
  orientation: 'vertical' | 'horizontal',
  collapsedSize?: number
) => {
  if (orientation === 'vertical') {
    return panel.height <= (collapsedSize ?? panel.minimumHeight);
  }
  return panel.width <= (collapsedSize ?? panel.minimumWidth);
};

export const useCollapsibleGridviewPanel = (
  tab: TabName,
  rootPanelId: Exclude<keyof TabPanelApis, 'main'>,
  panelId: string,
  orientation: 'horizontal' | 'vertical',
  defaultSize: number,
  collapsedSize?: number
) => {
  const $isCollapsed = useState(() => atom(false))[0];
  const lastExpandedSizeRef = useRef<number>(0);
  const collapse = useCallback(() => {
    const api = panelRegistry.getTabPanelApis(tab)?.[rootPanelId];
    if (!api) {
      return;
    }
    const panel = api.getPanel(panelId);
    if (!panel) {
      return;
    }

    lastExpandedSizeRef.current = orientation === 'vertical' ? panel.height : panel.width;

    if (orientation === 'vertical') {
      panel.api.setSize({ height: collapsedSize ?? panel.minimumHeight });
    } else {
      panel.api.setSize({ width: collapsedSize ?? panel.minimumWidth });
    }
  }, [collapsedSize, orientation, panelId, rootPanelId, tab]);

  const expand = useCallback(() => {
    const api = panelRegistry.getTabPanelApis(tab)?.[rootPanelId];

    if (!api) {
      return;
    }
    const panel = api.getPanel(panelId);
    if (!panel) {
      return;
    }
    if (orientation === 'vertical') {
      panel.api.setSize({ height: lastExpandedSizeRef.current || defaultSize });
    } else {
      panel.api.setSize({ width: lastExpandedSizeRef.current || defaultSize });
    }
  }, [defaultSize, orientation, panelId, rootPanelId, tab]);

  const toggle = useCallback(() => {
    const api = panelRegistry.getTabPanelApis(tab)?.[rootPanelId];

    if (!api) {
      return;
    }
    const panel = api.getPanel(panelId);
    if (!panel) {
      return;
    }
    const isCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
    if (isCollapsed) {
      expand();
    } else {
      collapse();
    }
  }, [tab, rootPanelId, panelId, orientation, collapsedSize, expand, collapse]);

  useEffect(() => {
    const api = panelRegistry.getTabPanelApis(tab)?.[rootPanelId];

    if (!api) {
      return;
    }
    const panel = api.getPanel(panelId);
    if (!panel) {
      return;
    }

    lastExpandedSizeRef.current = orientation === 'vertical' ? panel.height : panel.width;

    const disposable = panel.api.onDidDimensionsChange(() => {
      const isCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
      $isCollapsed.set(isCollapsed);
    });

    return () => {
      disposable.dispose();
    };
  }, [$isCollapsed, collapsedSize, orientation, panelId, rootPanelId, tab]);

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
