import type {
  DockviewApi,
  GridviewApi,
  IDockviewPanel,
  IDockviewReactProps,
  IGridviewPanel,
  IGridviewReactProps,
} from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { UpscalingLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/UpscalingLaunchpadPanel';
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
import {
  AutoLayoutProvider,
  PanelHotkeysLogical,
  useAutoLayoutContext,
  withPanelContainer,
} from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { memo, useCallback, useRef, useState } from 'react';

import { panelRegistry } from './panel-registry/panelApiRegistry';
import {
  BOARD_PANEL_DEFAULT_HEIGHT_PX,
  BOARD_PANEL_MIN_HEIGHT_PX,
  BOARDS_PANEL_ID,
  DEFAULT_TAB_ID,
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
import { UpscalingTabLeftPanel } from './UpscalingTabLeftPanel';
import { useResizeMainPanelOnFirstVisit } from './use-on-first-visible';

const tabComponents = {
  [DEFAULT_TAB_ID]: TabWithoutCloseButton,
  [TAB_WITH_PROGRESS_INDICATOR_ID]: TabWithoutCloseButtonAndWithProgressIndicator,
  [TAB_WITH_LAUNCHPAD_ICON_ID]: TabWithLaunchpadIcon,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(UpscalingLaunchpadPanel),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
  [PROGRESS_PANEL_ID]: withPanelContainer(GenerationProgressPanel),
};

const initializeCenterPanelLayout = (tab: TabName, api: DockviewApi) => {
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

  return { launchpad, viewer } satisfies Record<string, IDockviewPanel>;
};

const MainPanel = memo(() => {
  const { tab } = useAutoLayoutContext();
  const onReady = useCallback<IDockviewReactProps['onReady']>(
    ({ api }) => {
      const panels = initializeCenterPanelLayout(tab, api);
      panels.launchpad.api.setActive();
      panelRegistry.registerPanel(tab, 'main', api);

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

export const initializeRightPanelLayout = (tab: TabName, api: GridviewApi) => {
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

  boards.api.setSize({ height: BOARD_PANEL_DEFAULT_HEIGHT_PX, width: RIGHT_PANEL_MIN_SIZE_PX });

  return { gallery, boards } satisfies Record<string, IGridviewPanel>;
};

const RightPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeRightPanelLayout(tab, api);
      panelRegistry.registerPanel(tab, 'right', api);
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
  [SETTINGS_PANEL_ID]: withPanelContainer(UpscalingTabLeftPanel),
};

export const initializeLeftPanelLayout = (tab: TabName, api: GridviewApi) => {
  const settings = api.addPanel<PanelParameters>({
    id: SETTINGS_PANEL_ID,
    component: SETTINGS_PANEL_ID,
    params: {
      tab,
      focusRegion: 'settings',
    },
  });

  return { settings } satisfies Record<string, IGridviewPanel>;
};

const LeftPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeLeftPanelLayout(tab, api);
      panelRegistry.registerPanel(tab, 'left', api);
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

export const rootPanelComponents: RootLayoutGridviewComponents = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

export const initializeRootPanelLayout = (layoutApi: GridviewApi) => {
  const main = layoutApi.addPanel({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    priority: LayoutPriority.High,
  });

  const left = layoutApi.addPanel({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: main.id,
    },
  });

  const right = layoutApi.addPanel({
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

  return { main, left, right } satisfies Record<string, IGridviewPanel>;
};

export const UpscalingTabAutoLayout = memo(() => {
  const rootRef = useRef<HTMLDivElement>(null);
  const [rootApi, setRootApi] = useState<GridviewApi | null>(null);
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    const { main } = initializeRootPanelLayout(api);
    panelRegistry.registerPanel('upscaling', 'root', api);
    main.api.setActive();
    setRootApi(api);
  }, []);
  useResizeMainPanelOnFirstVisit(rootApi, rootRef);

  return (
    <AutoLayoutProvider tab="upscaling" rootRef={rootRef}>
      <GridviewReact
        ref={rootRef}
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
UpscalingTabAutoLayout.displayName = 'UpscalingTabAutoLayout';
