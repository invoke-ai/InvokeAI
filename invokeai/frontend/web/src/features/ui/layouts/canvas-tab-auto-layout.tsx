import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { BottomGalleryPanel } from 'features/gallery/components/BottomGalleryPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingLayerPanelButtons } from 'features/ui/components/FloatingLayerPanelButtons';
import { FloatingCanvasLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
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
import { memo, useCallback, useEffect } from 'react';

import { CanvasTabLeftPanel } from './CanvasTabLeftPanel';
import { CanvasWorkspacePanel } from './CanvasWorkspacePanel';
import { DockviewTabCanvasViewer } from './DockviewTabCanvasViewer';
import { DockviewTabCanvasWorkspace } from './DockviewTabCanvasWorkspace';
import { DockviewTabLaunchpad } from './DockviewTabLaunchpad';
import { navigationApi } from './navigation-api';
import { PanelHotkeysLogical } from './PanelHotkeysLogical';
import {
  BOTTOM_GALLERY_DEFAULT_HEIGHT_PX,
  BOTTOM_GALLERY_MIN_HEIGHT_PX,
  BOTTOM_GALLERY_PANEL_ID,
  DOCKVIEW_TAB_CANVAS_VIEWER_ID,
  DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
  DOCKVIEW_TAB_LAUNCHPAD_ID,
  LAUNCHPAD_PANEL_ID,
  LAYERS_PANEL_ID,
  LAYERS_PANEL_MIN_HEIGHT_PX,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  TOP_AREA_ID,
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
  [WORKSPACE_PANEL_ID]: withPanelContainer(CanvasWorkspacePanel),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
};

const initializeCenterPanelLayout = (tab: TabName, api: DockviewApi) => {
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

    api.addPanel<DockviewPanelParameters>({
      id: WORKSPACE_PANEL_ID,
      component: WORKSPACE_PANEL_ID,
      title: t('ui.panels.canvas'),
      tabComponent: DOCKVIEW_TAB_CANVAS_WORKSPACE_ID,
      params: {
        tab,
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

    launchpad.api.setActive();
  });
};

const MainPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IDockviewReactProps['onReady']>(
    ({ api }) => {
      initializeCenterPanelLayout(tab, api);
    },
    [tab]
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
      <FloatingLayerPanelButtons />
      <PanelHotkeysLogical />
    </>
  );
});
MainPanel.displayName = 'MainPanel';

/**
 * Right panel on canvas tab contains only the layers panel (no gallery/boards).
 */
const rightPanelComponents: AutoLayoutGridviewComponents = {
  [LAYERS_PANEL_ID]: withPanelContainer(CanvasLayersPanel),
};

const initializeRightPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'right', api, () => {
    api.addPanel<GridviewPanelParameters>({
      id: LAYERS_PANEL_ID,
      component: LAYERS_PANEL_ID,
      minimumHeight: LAYERS_PANEL_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'layers',
      },
    });
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

/**
 * Top area for canvas: left panel + main panel + right panel (layers only) side by side.
 */
const topAreaComponents: RootLayoutGridviewComponents = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

const initializeTopAreaLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'top-area', api, () => {
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

const TopArea = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeTopAreaLayout(tab, api);
    },
    [tab]
  );
  return (
    <GridviewReact
      className="dockview-theme-invoke"
      components={topAreaComponents}
      onReady={onReady}
      orientation={Orientation.VERTICAL}
    />
  );
});
TopArea.displayName = 'TopArea';

const bottomGalleryComponents: AutoLayoutGridviewComponents = {
  [BOTTOM_GALLERY_PANEL_ID]: withPanelContainer(BottomGalleryPanel),
};

const rootPanelComponents: RootLayoutGridviewComponents = {
  [TOP_AREA_ID]: TopArea,
  [BOTTOM_GALLERY_PANEL_ID]: bottomGalleryComponents[BOTTOM_GALLERY_PANEL_ID]!,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    const topArea = api.addPanel<GridviewPanelParameters>({
      id: TOP_AREA_ID,
      component: TOP_AREA_ID,
      priority: LayoutPriority.High,
    });

    const bottomGallery = api.addPanel<GridviewPanelParameters>({
      id: BOTTOM_GALLERY_PANEL_ID,
      component: BOTTOM_GALLERY_PANEL_ID,
      minimumHeight: BOTTOM_GALLERY_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'gallery',
      },
      position: {
        direction: 'below',
        referencePanel: topArea.id,
      },
    });

    bottomGallery.api.setSize({ height: BOTTOM_GALLERY_DEFAULT_HEIGHT_PX });
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
        orientation={Orientation.HORIZONTAL}
      />
    </AutoLayoutProvider>
  );
});
CanvasTabAutoLayout.displayName = 'CanvasTabAutoLayout';
