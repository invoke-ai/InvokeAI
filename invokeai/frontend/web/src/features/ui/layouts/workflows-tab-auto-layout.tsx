import type {
  DockviewApi,
  GridviewApi,
  IDockviewPanel,
  IDockviewReactProps,
  IGridviewPanel,
  IGridviewReactProps,
} from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { WorkflowsLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/WorkflowsLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
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
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { PanelHotkeysLogical } from './PanelHotkeysLogical';
import {
  BOARD_PANEL_DEFAULT_HEIGHT_PX,
  BOARD_PANEL_MIN_HEIGHT_PX,
  BOARDS_PANEL_ID,
  DEFAULT_TAB_ID,
  GALLERY_PANEL_DEFAULT_HEIGHT_PX,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_HEIGHT_PX,
  LAUNCHPAD_PANEL_ID,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  PROGRESS_PANEL_ID,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  TAB_WITH_LAUNCHPAD_ICON_ID,
  TAB_WITH_PROGRESS_INDICATOR_ID,
  VIEWER_PANEL_ID,
  WORKSPACE_PANEL_ID,
} from './shared';
import { TabWithLaunchpadIcon } from './TabWithLaunchpadIcon';
import { TabWithoutCloseButtonAndWithProgressIndicator } from './TabWithoutCloseButtonAndWithProgressIndicator';

const tabComponents = {
  [DEFAULT_TAB_ID]: TabWithoutCloseButton,
  [TAB_WITH_PROGRESS_INDICATOR_ID]: TabWithoutCloseButtonAndWithProgressIndicator,
  [TAB_WITH_LAUNCHPAD_ICON_ID]: TabWithLaunchpadIcon,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(WorkflowsLaunchpadPanel),
  [WORKSPACE_PANEL_ID]: withPanelContainer(NodeEditor),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
  [PROGRESS_PANEL_ID]: withPanelContainer(GenerationProgressPanel),
};

const initializeMainPanelLayout = (tab: TabName, api: DockviewApi) => {
  const launchpad = api.addPanel<PanelParameters>({
    id: LAUNCHPAD_PANEL_ID,
    component: LAUNCHPAD_PANEL_ID,
    title: 'Launchpad',
    tabComponent: TAB_WITH_LAUNCHPAD_ICON_ID,
    params: {
      tab,
      focusRegion: 'launchpad',
    },
  });

  const workspace = api.addPanel<PanelParameters>({
    id: WORKSPACE_PANEL_ID,
    component: WORKSPACE_PANEL_ID,
    title: 'Workflow Editor',
    tabComponent: DEFAULT_TAB_ID,
    params: {
      tab,
      focusRegion: 'workflows',
    },
    position: {
      direction: 'within',
      referencePanel: launchpad.id,
    },
  });

  const viewer = api.addPanel<PanelParameters>({
    id: VIEWER_PANEL_ID,
    component: VIEWER_PANEL_ID,
    title: 'Image Viewer',
    tabComponent: TAB_WITH_PROGRESS_INDICATOR_ID,
    params: {
      tab,
      focusRegion: 'viewer',
    },
    position: {
      direction: 'within',
      referencePanel: launchpad.id,
    },
  });

  // Register panels with navigation API
  navigationApi.registerPanel(tab, LAUNCHPAD_PANEL_ID, launchpad);
  navigationApi.registerPanel(tab, WORKSPACE_PANEL_ID, workspace);
  navigationApi.registerPanel(tab, VIEWER_PANEL_ID, viewer);

  return { launchpad, workspace, viewer } satisfies Record<string, IDockviewPanel>;
};

const MainPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IDockviewReactProps['onReady']>(
    ({ api }) => {
      const panels = initializeMainPanelLayout(tab, api);

      // Get the active panel from the navigation API's persisted state
      const activePanelId = navigationApi.getActiveTabMainPanel(tab);
      const activePanel = activePanelId ? panels[activePanelId as keyof typeof panels] : panels.launchpad;

      // Set the active panel (default to launchpad if no persisted state)
      activePanel?.api.setActive();

      const disposables = [
        api.onWillShowOverlay((e) => {
          if (e.kind === 'header_space' || e.kind === 'tab') {
            return;
          }
          e.preventDefault();
        }),
      ];

      return () => {
        disposables.forEach((disposable) => {
          disposable.dispose();
        });
      };
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

  // Check if there's persisted state for the gallery panel before setting default size
  const galleryPanelKey = `${tab}:${GALLERY_PANEL_ID}`;
  const savedGalleryState = navigationApi.getGridviewPanelState?.(galleryPanelKey);

  // Only set default size if there's no persisted height
  if (!savedGalleryState?.height) {
    gallery.api.setSize({ height: GALLERY_PANEL_DEFAULT_HEIGHT_PX, width: RIGHT_PANEL_MIN_SIZE_PX });
  } else {
    // Set only width if height is persisted
    gallery.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  }

  // Check if there's persisted state for the boards panel before setting default size
  const boardsPanelKey = `${tab}:${BOARDS_PANEL_ID}`;
  const savedBoardsState = navigationApi.getGridviewPanelState?.(boardsPanelKey);

  // Only set default size if there's no persisted height
  if (!savedBoardsState?.height) {
    boards.api.setSize({ height: BOARD_PANEL_DEFAULT_HEIGHT_PX, width: RIGHT_PANEL_MIN_SIZE_PX });
  } else {
    // Set only width if height is persisted
    boards.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  }

  // Register panels with navigation API
  navigationApi.registerPanel(tab, GALLERY_PANEL_ID, gallery);
  navigationApi.registerPanel(tab, BOARDS_PANEL_ID, boards);

  return { gallery, boards } satisfies Record<string, IGridviewPanel>;
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
  const settings = api.addPanel<PanelParameters>({
    id: SETTINGS_PANEL_ID,
    component: SETTINGS_PANEL_ID,
    params: {
      tab,
      focusRegion: 'settings',
    },
  });

  // Register panel with navigation API
  navigationApi.registerPanel(tab, SETTINGS_PANEL_ID, settings);

  return { settings } satisfies Record<string, IGridviewPanel>;
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

const initializeRootPanelLayout = (api: GridviewApi) => {
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
      referencePanel: MAIN_PANEL_ID,
    },
  });
  const right = api.addPanel({
    id: RIGHT_PANEL_ID,
    component: RIGHT_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: MAIN_PANEL_ID,
    },
  });

  left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  right.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });

  navigationApi.registerPanel('workflows', LEFT_PANEL_ID, left);
  navigationApi.registerPanel('workflows', MAIN_PANEL_ID, main);
  navigationApi.registerPanel('workflows', RIGHT_PANEL_ID, right);

  return { main, left, right } satisfies Record<string, IGridviewPanel>;
};

export const WorkflowsTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout(api);
    navigationApi.onTabReady('workflows');
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
