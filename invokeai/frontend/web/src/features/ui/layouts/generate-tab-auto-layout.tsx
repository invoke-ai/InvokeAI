import type {
  DockviewApi,
  GridviewApi,
  IDockviewPanel,
  IDockviewReactProps,
  IGridviewPanel,
  IGridviewReactProps,
} from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { GenerateLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/GenerateLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
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

import { GenerateTabLeftPanel } from './GenerateTabLeftPanel';
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
} from './shared';
import { TabWithLaunchpadIcon } from './TabWithLaunchpadIcon';
import { TabWithoutCloseButtonAndWithProgressIndicator } from './TabWithoutCloseButtonAndWithProgressIndicator';

const tabComponents = {
  [DEFAULT_TAB_ID]: TabWithoutCloseButton,
  [TAB_WITH_PROGRESS_INDICATOR_ID]: TabWithoutCloseButtonAndWithProgressIndicator,
  [TAB_WITH_LAUNCHPAD_ICON_ID]: TabWithLaunchpadIcon,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(GenerateLaunchpadPanel),
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
  navigationApi.registerPanel(tab, LAUNCHPAD_PANEL_ID, launchpad, {
    isActive: true,
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
  navigationApi.registerPanel(tab, VIEWER_PANEL_ID, viewer);

  return { launchpad, viewer } satisfies Record<string, IDockviewPanel>;
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
  navigationApi.registerPanel(tab, GALLERY_PANEL_ID, gallery, {
    dimensions: {
      height: GALLERY_PANEL_DEFAULT_HEIGHT_PX,
      width: RIGHT_PANEL_MIN_SIZE_PX,
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
  navigationApi.registerPanel(tab, BOARDS_PANEL_ID, boards, {
    dimensions: {
      height: BOARD_PANEL_DEFAULT_HEIGHT_PX,
      width: RIGHT_PANEL_MIN_SIZE_PX,
    },
  });

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
  [SETTINGS_PANEL_ID]: withPanelContainer(GenerateTabLeftPanel),
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

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  const main = api.addPanel<PanelParameters>({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    priority: LayoutPriority.High,
  });
  navigationApi.registerPanel(tab, MAIN_PANEL_ID, main);

  const left = api.addPanel<PanelParameters>({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: main.id,
    },
  });
  navigationApi.registerPanel(tab, LEFT_PANEL_ID, left, {
    dimensions: {
      width: LEFT_PANEL_MIN_SIZE_PX,
    },
  });

  const right = api.addPanel<PanelParameters>({
    id: RIGHT_PANEL_ID,
    component: RIGHT_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: main.id,
    },
  });
  navigationApi.registerPanel(tab, RIGHT_PANEL_ID, right, {
    dimensions: {
      width: RIGHT_PANEL_MIN_SIZE_PX,
    },
  });

  return { main, left, right } satisfies Record<string, IGridviewPanel>;
};

export const GenerateTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('generate', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('generate');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="generate">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
GenerateTabAutoLayout.displayName = 'GenerateTabAutoLayout';
