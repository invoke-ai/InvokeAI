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
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useRef, useState } from 'react';

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

const centerComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(UpscalingLaunchpadPanel),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
  [PROGRESS_PANEL_ID]: withPanelContainer(GenerationProgressPanel),
};

const initializeCenterPanelLayout = (api: DockviewApi) => {
  const launchpadPanel = api.addPanel<PanelParameters>({
    id: LAUNCHPAD_PANEL_ID,
    component: LAUNCHPAD_PANEL_ID,
    title: 'Launchpad',
    tabComponent: TAB_WITH_LAUNCHPAD_ICON_ID,
    params: {
      focusRegion: 'launchpad',
    },
  });

  const viewerPanel = api.addPanel<PanelParameters>({
    id: VIEWER_PANEL_ID,
    component: VIEWER_PANEL_ID,
    title: 'Image Viewer',
    tabComponent: TAB_WITH_PROGRESS_INDICATOR_ID,
    params: {
      focusRegion: 'viewer',
    },
    position: {
      direction: 'within',
      referencePanel: launchpadPanel.id,
    },
  });

  return { launchpadPanel, viewerPanel } satisfies Record<string, IDockviewPanel>;
};

const CenterPanel = memo(() => {
  const ctx = useAutoLayoutContext();
  const onReady = useCallback<IDockviewReactProps['onReady']>(
    (event) => {
      const panels = initializeCenterPanelLayout(event.api);
      panels.launchpadPanel.api.setActive();

      ctx._$centerPanelApi.set(event.api);

      const disposables = [
        event.api.onWillShowOverlay((e) => {
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
    [ctx._$centerPanelApi]
  );
  return (
    <>
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        tabComponents={tabComponents}
        components={centerComponents}
        onReady={onReady}
        theme={dockviewTheme}
      />
      <FloatingLeftPanelButtons />
      <FloatingRightPanelButtons />
      <PanelHotkeysLogical />
    </>
  );
});
CenterPanel.displayName = 'CenterPanel';

const rightPanelComponents: AutoLayoutGridviewComponents = {
  [BOARDS_PANEL_ID]: withPanelContainer(BoardsPanel),
  [GALLERY_PANEL_ID]: withPanelContainer(GalleryPanel),
};

export const initializeRightPanelLayout = (api: GridviewApi) => {
  const galleryPanel = api.addPanel<PanelParameters>({
    id: GALLERY_PANEL_ID,
    component: GALLERY_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: GALLERY_PANEL_MIN_HEIGHT_PX,
    params: {
      focusRegion: 'gallery',
    },
  });

  const boardsPanel = api.addPanel<PanelParameters>({
    id: BOARDS_PANEL_ID,
    component: BOARDS_PANEL_ID,
    minimumHeight: BOARD_PANEL_MIN_HEIGHT_PX,
    params: {
      focusRegion: 'boards',
    },
    position: {
      direction: 'above',
      referencePanel: galleryPanel.id,
    },
  });

  boardsPanel.api.setSize({ height: BOARD_PANEL_DEFAULT_HEIGHT_PX, width: RIGHT_PANEL_MIN_SIZE_PX });

  return { galleryPanel, boardsPanel } satisfies Record<string, IGridviewPanel>;
};

const onReadyRightPanel: IGridviewReactProps['onReady'] = (event) => {
  initializeRightPanelLayout(event.api);
};

const RightPanel = memo(() => {
  return (
    <>
      <GridviewReact
        className="dockview-theme-invoke"
        orientation={Orientation.VERTICAL}
        components={rightPanelComponents}
        onReady={onReadyRightPanel}
      />
    </>
  );
});
RightPanel.displayName = 'RightPanel';

const leftPanelComponents: AutoLayoutGridviewComponents = {
  [SETTINGS_PANEL_ID]: withPanelContainer(UpscalingTabLeftPanel),
};

export const initializeLeftPanelLayout = (api: GridviewApi) => {
  const settingsPanel = api.addPanel<PanelParameters>({
    id: SETTINGS_PANEL_ID,
    component: SETTINGS_PANEL_ID,
    params: {
      focusRegion: 'settings',
    },
  });

  return { settingsPanel } satisfies Record<string, IGridviewPanel>;
};

const LeftPanel = memo(() => {
  const ctx = useAutoLayoutContext();
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      initializeLeftPanelLayout(event.api);
      ctx._$leftPanelApi.set(event.api);
    },
    [ctx._$leftPanelApi]
  );
  return (
    <>
      <GridviewReact
        className="dockview-theme-invoke"
        orientation={Orientation.VERTICAL}
        components={leftPanelComponents}
        onReady={onReady}
      />
    </>
  );
});
LeftPanel.displayName = 'LeftPanel';

export const rootPanelComponents: RootLayoutGridviewComponents = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: CenterPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

export const initializeRootPanelLayout = (layoutApi: GridviewApi) => {
  const mainPanel = layoutApi.addPanel({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    priority: LayoutPriority.High,
  });

  const leftPanel = layoutApi.addPanel({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: mainPanel.id,
    },
  });

  const rightPanel = layoutApi.addPanel({
    id: RIGHT_PANEL_ID,
    component: RIGHT_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: mainPanel.id,
    },
  });

  leftPanel.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  rightPanel.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });

  return { mainPanel, leftPanel, rightPanel } satisfies Record<string, IGridviewPanel>;
};

export const UpscalingTabAutoLayout = memo(() => {
  const rootRef = useRef<HTMLDivElement>(null);
  const $rootPanelApi = useState(() => atom<GridviewApi | null>(null))[0];
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      $rootPanelApi.set(event.api);
      const { mainPanel } = initializeRootPanelLayout(event.api);
      mainPanel.api.setActive();
    },
    [$rootPanelApi]
  );
  useResizeMainPanelOnFirstVisit($rootPanelApi, rootRef);

  return (
    <AutoLayoutProvider $rootApi={$rootPanelApi} rootRef={rootRef} tab="upscaling">
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
