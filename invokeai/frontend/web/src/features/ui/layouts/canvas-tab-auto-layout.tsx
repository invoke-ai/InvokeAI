import type {
  DockviewApi,
  GridviewApi,
  IDockviewPanel,
  IDockviewReactProps,
  IGridviewPanel,
  IGridviewReactProps,
} from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/CanvasLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingCanvasLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import { AutoLayoutProvider, PanelHotkeysLogical, useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useRef, useState } from 'react';

import { CanvasTabLeftPanel } from './CanvasTabLeftPanel';
import { CanvasWorkspacePanel } from './CanvasWorkspacePanel';
import { registerFocusListener } from './layout-focus-bridge';
import {
  BOARD_PANEL_DEFAULT_HEIGHT_PX,
  BOARD_PANEL_MIN_HEIGHT_PX,
  BOARDS_PANEL_ID,
  DEFAULT_TAB_ID,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_HEIGHT_PX,
  LAUNCHPAD_PANEL_ID,
  LAYERS_PANEL_ID,
  LAYERS_PANEL_MIN_HEIGHT_PX,
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
import { useResizeMainPanelOnFirstVisit } from './use-on-first-visible';

const tabComponents = {
  [DEFAULT_TAB_ID]: TabWithoutCloseButton,
  [TAB_WITH_PROGRESS_INDICATOR_ID]: TabWithoutCloseButtonAndWithProgressIndicator,
  [TAB_WITH_LAUNCHPAD_ICON_ID]: TabWithLaunchpadIcon,
};

const centerPanelComponents: IDockviewReactProps['components'] = {
  [LAUNCHPAD_PANEL_ID]: CanvasLaunchpadPanel,
  [WORKSPACE_PANEL_ID]: CanvasWorkspacePanel,
  [VIEWER_PANEL_ID]: ImageViewerPanel,
  [PROGRESS_PANEL_ID]: GenerationProgressPanel,
};

const initializeCenterPanelLayout = (api: DockviewApi) => {
  const launchpadPanel = api.addPanel({
    id: LAUNCHPAD_PANEL_ID,
    component: LAUNCHPAD_PANEL_ID,
    title: 'Launchpad',
    tabComponent: TAB_WITH_LAUNCHPAD_ICON_ID,
  });
  registerFocusListener(launchpadPanel, 'launchpad');

  const workspacePanel = api.addPanel({
    id: WORKSPACE_PANEL_ID,
    component: WORKSPACE_PANEL_ID,
    title: 'Canvas',
    tabComponent: DEFAULT_TAB_ID,
    position: {
      direction: 'within',
      referencePanel: launchpadPanel.id,
    },
  });
  registerFocusListener(workspacePanel, 'canvas');

  const viewerPanel = api.addPanel({
    id: VIEWER_PANEL_ID,
    component: VIEWER_PANEL_ID,
    title: 'Image Viewer',
    tabComponent: DEFAULT_TAB_ID,
    position: {
      direction: 'within',
      referencePanel: launchpadPanel.id,
    },
  });
  registerFocusListener(viewerPanel, 'viewer');

  return { launchpadPanel, workspacePanel, viewerPanel } satisfies Record<string, IDockviewPanel>;
};

const CenterPanel = memo(() => {
  const ctx = useAutoLayoutContext();
  const onReady = useCallback<IDockviewReactProps['onReady']>(
    (event) => {
      const panels = initializeCenterPanelLayout(event.api);
      ctx._$centerPanelApi.set(event.api);

      panels.launchpadPanel.api.setActive();

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
        components={centerPanelComponents}
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
CenterPanel.displayName = 'CenterPanel';

const rightPanelComponents: IGridviewReactProps['components'] = {
  [BOARDS_PANEL_ID]: BoardsPanel,
  [GALLERY_PANEL_ID]: GalleryPanel,
  [LAYERS_PANEL_ID]: CanvasLayersPanel,
};

export const initializeRightPanelLayout = (api: GridviewApi) => {
  const galleryPanel = api.addPanel({
    id: GALLERY_PANEL_ID,
    component: GALLERY_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: GALLERY_PANEL_MIN_HEIGHT_PX,
  });
  registerFocusListener(galleryPanel, 'gallery');

  const layersPanel = api.addPanel({
    id: LAYERS_PANEL_ID,
    component: LAYERS_PANEL_ID,
    minimumHeight: LAYERS_PANEL_MIN_HEIGHT_PX,
    position: {
      direction: 'below',
      referencePanel: galleryPanel.id,
    },
  });
  registerFocusListener(layersPanel, 'layers');

  const boardsPanel = api.addPanel({
    id: BOARDS_PANEL_ID,
    component: BOARDS_PANEL_ID,
    minimumHeight: BOARD_PANEL_MIN_HEIGHT_PX,
    position: {
      direction: 'above',
      referencePanel: galleryPanel.id,
    },
  });
  registerFocusListener(boardsPanel, 'boards');

  boardsPanel.api.setSize({ height: BOARD_PANEL_DEFAULT_HEIGHT_PX, width: RIGHT_PANEL_MIN_SIZE_PX });
  return { galleryPanel, layersPanel, boardsPanel } satisfies Record<string, IGridviewPanel>;
};

const RightPanel = memo(() => {
  const ctx = useAutoLayoutContext();
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      initializeRightPanelLayout(event.api);
      ctx._$rightPanelApi.set(event.api);
    },
    [ctx._$rightPanelApi]
  );
  return (
    <>
      <GridviewReact
        className="dockview-theme-invoke"
        orientation={Orientation.VERTICAL}
        components={rightPanelComponents}
        onReady={onReady}
      />
    </>
  );
});
RightPanel.displayName = 'RightPanel';

const leftPanelComponents: IGridviewReactProps['components'] = {
  [SETTINGS_PANEL_ID]: CanvasTabLeftPanel,
};

export const initializeLeftPanelLayout = (api: GridviewApi) => {
  const settingsPanel = api.addPanel({
    id: SETTINGS_PANEL_ID,
    component: SETTINGS_PANEL_ID,
  });
  registerFocusListener(settingsPanel, 'settings');

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

export const rootPanelComponents: IGridviewReactProps['components'] = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: CenterPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

export const initializeRootPanelLayout = (api: GridviewApi) => {
  const mainPanel = api.addPanel({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    priority: LayoutPriority.High,
  });

  const leftPanel = api.addPanel({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: mainPanel.id,
    },
  });

  const rightPanel = api.addPanel({
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

export const CanvasTabAutoLayout = memo(() => {
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
    <AutoLayoutProvider $rootApi={$rootPanelApi} rootRef={rootRef} tab="canvas">
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
CanvasTabAutoLayout.displayName = 'CanvasTabAutoLayout';
