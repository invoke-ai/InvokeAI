import { useAppDispatch } from 'app/store/storeHooks';
import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { activeCanvasChanged,canvasInstanceAdded } from 'features/controlLayers/store/canvasesSlice';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingCanvasLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import type {
  AutoLayoutDockviewComponents,
  AutoLayoutGridviewComponents,
  DockviewPanelParameters,
  GridviewPanelParameters,
  RootLayoutGridviewComponents,
} from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider, useAutoLayoutContext, withPanelContainer } from 'features/ui/layouts/auto-layout-context';
import { CanvasLaunchpadPanel } from 'features/ui/layouts/CanvasLaunchpadPanel';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { t } from 'i18next';
import { nanoid } from 'nanoid';
import { memo, useCallback, useEffect } from 'react';

import { CanvasTabLeftPanel } from './CanvasTabLeftPanel';
import { CanvasWorkspacePanel } from './CanvasWorkspacePanel';
import { DockviewTabCanvasViewer } from './DockviewTabCanvasViewer';
import { DockviewTabCanvasWorkspace } from './DockviewTabCanvasWorkspace';
import { DockviewTabLaunchpad } from './DockviewTabLaunchpad';
import { navigationApi } from './navigation-api';
import { PanelHotkeysLogical } from './PanelHotkeysLogical';
import {
  BOARD_PANEL_MIN_HEIGHT_PX,
  BOARDS_PANEL_ID,
  CANVAS_BOARD_PANEL_DEFAULT_HEIGHT_PX,
  DOCKVIEW_TAB_CANVAS_VIEWER_ID,
  DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
  DOCKVIEW_TAB_LAUNCHPAD_ID,
  GALLERY_PANEL_DEFAULT_HEIGHT_PX,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_HEIGHT_PX,
  LAUNCHPAD_PANEL_ID,
  LAYERS_PANEL_ID,
  LAYERS_PANEL_MIN_HEIGHT_PX,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  VIEWER_PANEL_ID,
  WORKSPACE_PANEL_ID,
} from './shared';

const tabComponents = {
  [DOCKVIEW_TAB_LAUNCHPAD_ID]: DockviewTabLaunchpad,
  [DOCKVIEW_TAB_CANVAS_VIEWER_ID]: DockviewTabCanvasViewer,
  [DOCKVIEW_TAB_CANVAS_WORKSPACE_ID]: DockviewTabCanvasWorkspace,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(CanvasLaunchpadPanel),
  [WORKSPACE_PANEL_ID]: CanvasWorkspacePanel, // Custom wrapper that handles props
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
};

const initializeCenterPanelLayout = (tab: TabName, api: DockviewApi, dispatch: ReturnType<typeof useAppDispatch>) => {
  navigationApi.registerContainer(tab, 'main', api, () => {
    const launchpad = api.addPanel<DockviewPanelParameters>({
      id: LAUNCHPAD_PANEL_ID,
      component: LAUNCHPAD_PANEL_ID,
      title: t('ui.panels.launchpad'),
      tabComponent: DOCKVIEW_TAB_LAUNCHPAD_ID,
      params: {
        tab,
        focusRegion: 'launchpad',
        i18nKey: 'ui.panels.launchpad',
      },
    });

    // Create first canvas instance
    const firstCanvasId = nanoid();
    dispatch(canvasInstanceAdded({ canvasId: firstCanvasId, name: 'Canvas 1' }));
    // Set it as the active canvas
    dispatch(activeCanvasChanged({ canvasId: firstCanvasId }));
    
    const canvasPanel = api.addPanel<DockviewPanelParameters>({
      id: `${WORKSPACE_PANEL_ID}_${firstCanvasId}`,
      component: WORKSPACE_PANEL_ID,
      title: 'Canvas 1',
      tabComponent: DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
      params: {
        tab,
        canvasId: firstCanvasId,
        focusRegion: 'canvas',
        i18nKey: 'ui.panels.canvas',
      },
      position: {
        direction: 'within',
        referencePanel: launchpad.id,
      },
    });

    api.addPanel<DockviewPanelParameters>({
      id: VIEWER_PANEL_ID,
      component: VIEWER_PANEL_ID,
      title: t('ui.panels.imageViewer'),
      tabComponent: DOCKVIEW_TAB_CANVAS_VIEWER_ID,
      params: {
        tab,
        focusRegion: 'viewer',
        i18nKey: 'ui.panels.imageViewer',
      },
      position: {
        direction: 'within',
        referencePanel: launchpad.id,
      },
    });

    // Set the canvas panel as the active panel (not launchpad)
    canvasPanel.api.setActive();

    // Track active canvas panel changes
    api.onDidActivePanelChange((panel) => {
      console.log('Panel activated:', panel?.id, 'params:', panel?.params);
      if (panel?.id.startsWith(WORKSPACE_PANEL_ID)) {
        const canvasId = panel.params?.canvasId;
        if (canvasId) {
          console.log('Setting active canvas:', canvasId);
          dispatch(activeCanvasChanged({ canvasId }));
        }
      } else {
        // When a non-canvas panel is activated, set active canvas to null
        console.log('Setting active canvas to null');
        dispatch(activeCanvasChanged({ canvasId: null }));
      }
    });

  });
};

const MainPanel = memo(() => {
  const { tab } = useAutoLayoutContext();
  const dispatch = useAppDispatch();

  const onReady = useCallback<IDockviewReactProps['onReady']>(
    ({ api }) => {
      initializeCenterPanelLayout(tab, api, dispatch);
    },
    [tab, dispatch]
  );
  return (
    <>
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        components={mainPanelComponents}
        onReady={onReady}
        theme={dockviewTheme}
        tabComponents={tabComponents}
      />
      <FloatingCanvasLeftPanelButtons />
      <FloatingRightPanelButtons />
      <PanelHotkeysLogical />
    </>
  );
});
MainPanel.displayName = 'MainPanel';

const rightPanelComponents: AutoLayoutGridviewComponents = {
  [BOARDS_PANEL_ID]: withPanelContainer(BoardsPanel),
  [GALLERY_PANEL_ID]: withPanelContainer(GalleryPanel),
  [LAYERS_PANEL_ID]: withPanelContainer(CanvasLayersPanel),
};

const initializeRightPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'right', api, () => {
    const gallery = api.addPanel<GridviewPanelParameters>({
      id: GALLERY_PANEL_ID,
      component: GALLERY_PANEL_ID,
      minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      minimumHeight: GALLERY_PANEL_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'gallery',
      },
    });

    const boards = api.addPanel<GridviewPanelParameters>({
      id: BOARDS_PANEL_ID,
      component: BOARDS_PANEL_ID,
      minimumHeight: BOARD_PANEL_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'boards',
      },
      position: {
        direction: 'above',
        referencePanel: gallery.id,
      },
    });

    api.addPanel<GridviewPanelParameters>({
      id: LAYERS_PANEL_ID,
      component: LAYERS_PANEL_ID,
      minimumHeight: LAYERS_PANEL_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'layers',
      },
      position: {
        direction: 'below',
        referencePanel: gallery.id,
      },
    });

    gallery.api.setSize({ height: GALLERY_PANEL_DEFAULT_HEIGHT_PX });
    boards.api.setSize({ height: CANVAS_BOARD_PANEL_DEFAULT_HEIGHT_PX });
  });
};

const RightPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeRightPanelLayout(tab, api);
    },
    [tab]
  );
  return (
    <GridviewReact
      className="dockview-theme-invoke"
      orientation={Orientation.VERTICAL}
      components={rightPanelComponents}
      onReady={onReady}
    />
  );
});
RightPanel.displayName = 'RightPanel';

const leftPanelComponents: AutoLayoutGridviewComponents = {
  [SETTINGS_PANEL_ID]: withPanelContainer(CanvasTabLeftPanel),
};

const initializeLeftPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'left', api, () => {
    api.addPanel<GridviewPanelParameters>({
      id: SETTINGS_PANEL_ID,
      component: SETTINGS_PANEL_ID,
      params: {
        tab,
        focusRegion: 'settings',
      },
    });
  });
};

const LeftPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeLeftPanelLayout(tab, api);
    },
    [tab]
  );
  return (
    <GridviewReact
      className="dockview-theme-invoke"
      orientation={Orientation.VERTICAL}
      components={leftPanelComponents}
      onReady={onReady}
    />
  );
});
LeftPanel.displayName = 'LeftPanel';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    const main = api.addPanel({
      id: MAIN_PANEL_ID,
      component: MAIN_PANEL_ID,
      priority: LayoutPriority.High,
    });

    const left = api.addPanel({
      id: LEFT_PANEL_ID,
      component: LEFT_PANEL_ID,
      minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      priority: LayoutPriority.Low,
      position: {
        direction: 'left',
        referencePanel: main.id,
      },
    });

    const right = api.addPanel({
      id: RIGHT_PANEL_ID,
      component: RIGHT_PANEL_ID,
      minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      priority: LayoutPriority.Low,
      position: {
        direction: 'right',
        referencePanel: main.id,
      },
    });

    left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
    right.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  });
};

export const CanvasTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('canvas', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('canvas');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="canvas">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
CanvasTabAutoLayout.displayName = 'CanvasTabAutoLayout';
