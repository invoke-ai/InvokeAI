import type { GridviewApi, GridviewPanelApi, IGridviewPanel } from 'dockview';
import { atom } from 'nanostores';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

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
  api: GridviewApi | null,
  panelId: string,
  orientation: 'horizontal' | 'vertical',
  defaultSize: number,
  collapsedSize?: number
) => {
  const $isCollapsed = useState(() => atom(false))[0];
  const lastExpandedSizeRef = useRef<number>(0);
  const retryCountRef = useRef<number>(0);
  const maxRetries = 10;

  const collapse = useCallback(() => {
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
  }, [api, collapsedSize, orientation, panelId]);

  const expand = useCallback(() => {
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
  }, [api, defaultSize, orientation, panelId]);

  const toggle = useCallback(() => {
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
  }, [api, panelId, orientation, collapsedSize, expand, collapse]);

  useEffect(() => {
    if (!api) {
      return;
    }
    
    const initializePanel = () => {
      const panel = api.getPanel(panelId);
      if (!panel) {
        retryCountRef.current++;
        if (retryCountRef.current >= maxRetries) {
          // Panel not available after max retries, stop trying
          return;
        }
        // Panel not available yet, retry after a short delay
        setTimeout(initializePanel, 10);
        return;
      }

      // Reset retry count on success
      retryCountRef.current = 0;

      // Set initial state immediately
      const initialIsCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
      $isCollapsed.set(initialIsCollapsed);
      
      // Initialize lastExpandedSizeRef properly
      if (initialIsCollapsed) {
        // If starting collapsed, use default size as the target expansion size
        lastExpandedSizeRef.current = defaultSize;
      } else {
        // If starting expanded, use current size as the target expansion size
        lastExpandedSizeRef.current = orientation === 'vertical' ? panel.height : panel.width;
      }

      const disposable = panel.api.onDidDimensionsChange(() => {
        const isCollapsed = getIsCollapsed(panel, orientation, collapsedSize);
        $isCollapsed.set(isCollapsed);
      });

      return () => {
        disposable.dispose();
      };
    };

    const cleanup = initializePanel();
    return cleanup;
  }, [api, collapsedSize, orientation, panelId, defaultSize, $isCollapsed]);

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
