import type { GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, Orientation } from 'dockview';
import { WorkflowsLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/WorkflowsLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import NodeEditor from 'features/nodes/components/NodeEditor';
import WorkflowsTabLeftPanel from 'features/nodes/components/sidePanel/WorkflowsTabLeftPanel';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/store/uiSlice';
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useRef, useState } from 'react';

import { useOnFirstVisible } from './use-on-first-visible';

const LAUNCHPAD_PANEL_ID = 'launchpad';
const WORKSPACE_PANEL_ID = 'workspace';
const VIEWER_PANEL_ID = 'viewer';
const PROGRESS_PANEL_ID = 'progress';

const dockviewComponents: IDockviewReactProps['components'] = {
  [LAUNCHPAD_PANEL_ID]: WorkflowsLaunchpadPanel,
  [WORKSPACE_PANEL_ID]: NodeEditor,
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
    title: 'Workflow Editor',
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
    <DockviewReact
      disableDnd={true}
      locked={true}
      disableFloatingGroups={true}
      dndEdges={false}
      defaultTabComponent={TabWithoutCloseButton}
      components={dockviewComponents}
      onReady={onReadyMainPanel}
      theme={dockviewTheme}
    />
  );
});
MainPanel.displayName = 'MainPanel';

const LEFT_PANEL_ID = 'left';
const MAIN_PANEL_ID = 'main';
const BOARDS_PANEL_ID = 'boards';
const GALLERY_PANEL_ID = 'gallery';

export const gridviewComponents: IGridviewReactProps['components'] = {
  [LEFT_PANEL_ID]: WorkflowsTabLeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
  [BOARDS_PANEL_ID]: BoardsPanel,
  [GALLERY_PANEL_ID]: GalleryPanel,
};

export const initializeLayout = (api: GridviewApi) => {
  api.addPanel({
    id: MAIN_PANEL_ID,
    component: MAIN_PANEL_ID,
    // priority: LayoutPriority.High,
  });
  api.addPanel({
    id: LEFT_PANEL_ID,
    component: LEFT_PANEL_ID,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: MAIN_PANEL_ID,
    },
    // priority: LayoutPriority.High,
  });
  api.addPanel({
    id: GALLERY_PANEL_ID,
    component: GALLERY_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: 232,
    position: {
      direction: 'right',
      referencePanel: MAIN_PANEL_ID,
    },
    // priority: LayoutPriority.High,
  });
  api.addPanel({
    id: BOARDS_PANEL_ID,
    component: BOARDS_PANEL_ID,
    minimumHeight: 36,
    position: {
      direction: 'above',
      referencePanel: GALLERY_PANEL_ID,
    },
    // priority: LayoutPriority.High,
  });
  api.getPanel(LEFT_PANEL_ID)?.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  api.getPanel(BOARDS_PANEL_ID)?.api.setSize({ height: 256, width: RIGHT_PANEL_MIN_SIZE_PX });
  api.getPanel(MAIN_PANEL_ID)?.api.setActive();
};

export const WorkflowsTabAutoLayout = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const $api = useState(() => atom<GridviewApi | null>(null))[0];
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      $api.set(event.api);
      initializeLayout(event.api);
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

  return (
    <AutoLayoutProvider $api={$api}>
      <GridviewReact
        ref={ref}
        className="dockview-theme-invoke"
        components={gridviewComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
WorkflowsTabAutoLayout.displayName = 'WorkflowsTabAutoLayout';
