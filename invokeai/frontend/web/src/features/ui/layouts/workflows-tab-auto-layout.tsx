import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import NodeEditor from 'features/nodes/components/NodeEditor';
import WorkflowsTabLeftPanel from 'features/nodes/components/sidePanel/WorkflowsTabLeftPanel';
import { FloatingLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import type {
  AutoLayoutDockviewComponents,
  AutoLayoutGridviewComponents,
  PanelParameters,
  RootLayoutGridviewComponents,
} from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider, useAutoLayoutContext, withPanelContainer } from 'features/ui/layouts/auto-layout-context';
import { DockviewTab } from 'features/ui/layouts/DockviewTab';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { memo, useCallback, useEffect } from 'react';

import { DockviewTabLaunchpad } from './DockviewTabLaunchpad';
import { DockviewTabProgress } from './DockviewTabProgress';
import { navigationApi } from './navigation-api';
import { PanelHotkeysLogical } from './PanelHotkeysLogical';
import {
  BOARD_PANEL_DEFAULT_HEIGHT_PX,
  BOARD_PANEL_MIN_HEIGHT_PX,
  BOARDS_PANEL_ID,
  DOCKVIEW_TAB_ID,
  DOCKVIEW_TAB_LAUNCHPAD_ID,
  DOCKVIEW_TAB_PROGRESS_ID,
  GALLERY_PANEL_DEFAULT_HEIGHT_PX,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_HEIGHT_PX,
  LAUNCHPAD_PANEL_ID,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  VIEWER_PANEL_ID,
  WORKSPACE_PANEL_ID,
} from './shared';
import { WorkflowsLaunchpadPanel } from './WorkflowsLaunchpadPanel';

const tabComponents = {
  [DOCKVIEW_TAB_ID]: DockviewTab,
  [DOCKVIEW_TAB_PROGRESS_ID]: DockviewTabProgress,
  [DOCKVIEW_TAB_LAUNCHPAD_ID]: DockviewTabLaunchpad,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(WorkflowsLaunchpadPanel),
  [WORKSPACE_PANEL_ID]: withPanelContainer(NodeEditor),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
};

const initializeMainPanelLayout = (tab: TabName, api: DockviewApi) => {
  navigationApi.registerContainer(tab, 'main', api, () => {
    const launchpad = api.addPanel<PanelParameters>({
      id: LAUNCHPAD_PANEL_ID,
      component: LAUNCHPAD_PANEL_ID,
      title: 'Launchpad',
      tabComponent: DOCKVIEW_TAB_LAUNCHPAD_ID,
      params: {
        tab,
        focusRegion: 'launchpad',
      },
    });

    api.addPanel<PanelParameters>({
      id: WORKSPACE_PANEL_ID,
      component: WORKSPACE_PANEL_ID,
      title: 'Workflow Editor',
      tabComponent: DOCKVIEW_TAB_ID,
      params: {
        tab,
        focusRegion: 'workflows',
      },
      position: {
        direction: 'within',
        referencePanel: launchpad.id,
      },
    });

    api.addPanel<PanelParameters>({
      id: VIEWER_PANEL_ID,
      component: VIEWER_PANEL_ID,
      title: 'Image Viewer',
      tabComponent: DOCKVIEW_TAB_PROGRESS_ID,
      params: {
        tab,
        focusRegion: 'viewer',
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
      initializeMainPanelLayout(tab, api);
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
        tabComponents={tabComponents}
        components={mainPanelComponents}
        onReady={onReady}
        theme={dockviewTheme}
      />
      <FloatingLeftPanelButtons />
      <FloatingRightPanelButtons />
      <PanelHotkeysLogical />
    </>
  );
});
MainPanel.displayName = 'MainPanel';

const rightPanelComponents: AutoLayoutGridviewComponents = {
  [BOARDS_PANEL_ID]: withPanelContainer(BoardsPanel),
  [GALLERY_PANEL_ID]: withPanelContainer(GalleryPanel),
};

const initializeRightPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'right', api, () => {
    const gallery = api.addPanel<PanelParameters>({
      id: GALLERY_PANEL_ID,
      component: GALLERY_PANEL_ID,
      minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      minimumHeight: GALLERY_PANEL_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'gallery',
      },
    });

    const boards = api.addPanel<PanelParameters>({
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

    gallery.api.setSize({ height: GALLERY_PANEL_DEFAULT_HEIGHT_PX });
    boards.api.setSize({ height: BOARD_PANEL_DEFAULT_HEIGHT_PX });
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
  [SETTINGS_PANEL_ID]: withPanelContainer(WorkflowsTabLeftPanel),
};

const initializeLeftPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'left', api, () => {
    api.addPanel<PanelParameters>({
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
      position: {
        direction: 'left',
        referencePanel: main.id,
      },
    });

    const right = api.addPanel({
      id: RIGHT_PANEL_ID,
      component: RIGHT_PANEL_ID,
      minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
      position: {
        direction: 'right',
        referencePanel: main.id,
      },
    });

    left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
    right.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  });
};

export const WorkflowsTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('workflows', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('workflows');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="workflows">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
WorkflowsTabAutoLayout.displayName = 'WorkflowsTabAutoLayout';
