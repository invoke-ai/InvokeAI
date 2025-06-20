import type { GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/CanvasLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/store/uiSlice';
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useRef, useState } from 'react';

import { CanvasTabLeftPanel } from './CanvasTabLeftPanel';
import { CanvasWorkspacePanel } from './CanvasWorkspacePanel';
import { useOnFirstVisible } from './use-on-first-visible';

const LAUNCHPAD_PANEL_ID = 'launchpad';
const WORKSPACE_PANEL_ID = 'workspace';
const VIEWER_PANEL_ID = 'viewer';
const PROGRESS_PANEL_ID = 'progress';

const mainPanelComponents: IDockviewReactProps['components'] = {
  [LAUNCHPAD_PANEL_ID]: CanvasLaunchpadPanel,
  [WORKSPACE_PANEL_ID]: CanvasWorkspacePanel,
  [VIEWER_PANEL_ID]: ImageViewerPanel,
  [PROGRESS_PANEL_ID]: GenerationProgressPanel,
};

const onReadyMainPanel: IDockviewReactProps['onReady'] = (event) => {
  const { api } = event;
  api.addPanel({
    id: LAUNCHPAD_PANEL_ID,
    component: LAUNCHPAD_PANEL_ID,
    title: 'Launchpad',
  });
  api.addPanel({
    id: WORKSPACE_PANEL_ID,
    component: WORKSPACE_PANEL_ID,
    title: 'Canvas',
    position: {
      direction: 'within',
      referencePanel: LAUNCHPAD_PANEL_ID,
    },
  });
  api.addPanel({
    id: VIEWER_PANEL_ID,
    component: VIEWER_PANEL_ID,
    title: 'Image Viewer',
    position: {
      direction: 'within',
      referencePanel: LAUNCHPAD_PANEL_ID,
    },
  });
  api.addPanel({
    id: PROGRESS_PANEL_ID,
    component: PROGRESS_PANEL_ID,
    title: 'Generation Progress',
    position: {
      direction: 'within',
      referencePanel: LAUNCHPAD_PANEL_ID,
    },
  });

  api.getPanel(LAUNCHPAD_PANEL_ID)?.api.setActive();

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
};

const MainPanel = memo(() => {
  return (
    <>
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        defaultTabComponent={TabWithoutCloseButton}
        components={mainPanelComponents}
        onReady={onReadyMainPanel}
        theme={dockviewTheme}
      />
      <FloatingLeftPanelButtons />
      <FloatingRightPanelButtons />
    </>
  );
});
MainPanel.displayName = 'MainPanel';

const BOARDS_PANEL_ID = 'boards';
const GALLERY_PANEL_ID = 'gallery';
const LAYERS_PANEL_ID = 'layers';

const rightPanelComponents: IGridviewReactProps['components'] = {
  [BOARDS_PANEL_ID]: BoardsPanel,
  [GALLERY_PANEL_ID]: GalleryPanel,
  [LAYERS_PANEL_ID]: CanvasLayersPanel,
};

export const initializeRightLayout = (api: GridviewApi) => {
  api.addPanel({
    id: GALLERY_PANEL_ID,
    component: GALLERY_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: 232,
  });
  api.addPanel({
    id: LAYERS_PANEL_ID,
    component: LAYERS_PANEL_ID,
    minimumHeight: 256,
    position: {
      direction: 'below',
      referencePanel: GALLERY_PANEL_ID,
    },
  });
  api.addPanel({
    id: BOARDS_PANEL_ID,
    component: BOARDS_PANEL_ID,
    minimumHeight: 36,
    position: {
      direction: 'above',
      referencePanel: GALLERY_PANEL_ID,
    },
  });
  api.getPanel(BOARDS_PANEL_ID)?.api.setSize({ height: 256, width: RIGHT_PANEL_MIN_SIZE_PX });
};

const onReadyRightPanel: IGridviewReactProps['onReady'] = (event) => {
  initializeRightLayout(event.api);
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

const LEFT_PANEL_ID = 'left';
const MAIN_PANEL_ID = 'main';
const RIGHT_PANEL_ID = 'right';

export const rootComponents: IGridviewReactProps['components'] = {
  [LEFT_PANEL_ID]: CanvasTabLeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
  [RIGHT_PANEL_ID]: RightPanel,
};

export const initializeRootLayout = (api: GridviewApi) => {
  api.addPanel({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    priority: LayoutPriority.High,
  });
  api.addPanel({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: MAIN_PANEL_ID,
    },
  });
  api.addPanel({
    id: RIGHT_PANEL_ID,
    component: RIGHT_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: MAIN_PANEL_ID,
    },
  });
  api.getPanel(LEFT_PANEL_ID)?.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  api.getPanel(RIGHT_PANEL_ID)?.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  api.getPanel(MAIN_PANEL_ID)?.api.setActive();
};

export const CanvasTabAutoLayout = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const $api = useState(() => atom<GridviewApi | null>(null))[0];
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      $api.set(event.api);
      initializeRootLayout(event.api);
    },
    [$api]
  );
  const resizeMainPanelOnFirstVisible = useCallback(() => {
    const api = $api.get();
    if (!api) {
      return;
    }
    const mainPanel = api.getPanel(MAIN_PANEL_ID);
    if (!mainPanel) {
      return;
    }
    if (mainPanel.width !== 0) {
      return;
    }
    let count = 0;
    const setSize = () => {
      if (count++ > 50) {
        return;
      }
      mainPanel.api.setSize({ width: Number.MAX_SAFE_INTEGER });
      if (mainPanel.width === 0) {
        requestAnimationFrame(setSize);
        return;
      }
    };
    setSize();
  }, [$api]);
  useOnFirstVisible(ref, resizeMainPanelOnFirstVisible);
  const toggleLeftPanel = useCallback(() => {
    const api = $api.get();
    if (!api) {
      return;
    }
    const left = api.getPanel(LEFT_PANEL_ID);
    if (!left) {
      return;
    }
    if (left.maximumWidth === 0) {
      left.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: LEFT_PANEL_MIN_SIZE_PX });
      left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
    } else {
      left.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
      left.api.setSize({ width: 0 });
    }
  }, [$api]);
  const toggleRightPanel = useCallback(() => {
    const api = $api.get();
    if (!api) {
      return;
    }
    const right = api.getPanel(RIGHT_PANEL_ID);
    if (!right) {
      return;
    }
    if (right.maximumWidth === 0) {
      right.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: RIGHT_PANEL_MIN_SIZE_PX });
      right.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
    } else {
      right.api.setConstraints({ maximumWidth: 0, minimumWidth: 0 });
      right.api.setSize({ width: 0 });
    }
  }, [$api]);

  return (
    <AutoLayoutProvider $api={$api} toggleLeftPanel={toggleLeftPanel} toggleRightPanel={toggleRightPanel}>
      <GridviewReact
        ref={ref}
        className="dockview-theme-invoke"
        components={rootComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
CanvasTabAutoLayout.displayName = 'CanvasTabAutoLayout';
