import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { UpscalingLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/UpscalingLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import { AutoLayoutProvider, PanelHotkeysLogical, useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useRef, useState } from 'react';

import {
  BOARDS_PANEL_ID,
  DEFAULT_TAB_ID,
  GALLERY_PANEL_ID,
  LAUNCHPAD_PANEL_ID,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  PROGRESS_PANEL_ID,
  RIGHT_PANEL_ID,
  RIGHT_PANEL_MIN_SIZE_PX,
  SETTINGS_PANEL_ID,
  TAB_WITH_PROGRESS_INDICATOR_ID,
  VIEWER_PANEL_ID,
} from './shared';
import { TabWithoutCloseButtonAndWithProgressIndicator } from './TabWithoutCloseButtonAndWithProgressIndicator';
import { UpscalingTabLeftPanel } from './UpscalingTabLeftPanel';
import { useResizeMainPanelOnFirstVisit } from './use-on-first-visible';

const tabComponents = {
  [DEFAULT_TAB_ID]: TabWithoutCloseButton,
  [TAB_WITH_PROGRESS_INDICATOR_ID]: TabWithoutCloseButtonAndWithProgressIndicator,
};

const centerComponents: IDockviewReactProps['components'] = {
  [LAUNCHPAD_PANEL_ID]: UpscalingLaunchpadPanel,
  [VIEWER_PANEL_ID]: ImageViewerPanel,
  [PROGRESS_PANEL_ID]: GenerationProgressPanel,
};

const initializeCenterLayout = (api: DockviewApi) => {
  api.addPanel({
    id: LAUNCHPAD_PANEL_ID,
    component: LAUNCHPAD_PANEL_ID,
    title: 'Launchpad',
    tabComponent: DEFAULT_TAB_ID,
  });
  api.addPanel({
    id: VIEWER_PANEL_ID,
    component: VIEWER_PANEL_ID,
    title: 'Image Viewer',
    tabComponent: TAB_WITH_PROGRESS_INDICATOR_ID,
    position: {
      direction: 'within',
      referencePanel: LAUNCHPAD_PANEL_ID,
    },
  });
  api.addPanel({
    id: PROGRESS_PANEL_ID,
    component: PROGRESS_PANEL_ID,
    title: 'Generation Progress',
    tabComponent: TAB_WITH_PROGRESS_INDICATOR_ID,
    position: {
      direction: 'within',
      referencePanel: LAUNCHPAD_PANEL_ID,
    },
  });

  api.getPanel(LAUNCHPAD_PANEL_ID)?.api.setActive();
};
const CenterPanel = memo(() => {
  const ctx = useAutoLayoutContext();
  const onReady = useCallback<IDockviewReactProps['onReady']>(
    (event) => {
      initializeCenterLayout(event.api);
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

const rightPanelComponents: IGridviewReactProps['components'] = {
  [BOARDS_PANEL_ID]: BoardsPanel,
  [GALLERY_PANEL_ID]: GalleryPanel,
};

export const initializeRightPanelLayout = (api: GridviewApi) => {
  api.addPanel({
    id: GALLERY_PANEL_ID,
    component: GALLERY_PANEL_ID,
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: 232,
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

const leftPanelComponents: IGridviewReactProps['components'] = {
  [SETTINGS_PANEL_ID]: UpscalingTabLeftPanel,
};

export const initializeLeftPanelLayout = (api: GridviewApi) => {
  api.addPanel({
    id: SETTINGS_PANEL_ID,
    component: SETTINGS_PANEL_ID,
  });
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

export const UpscalingTabAutoLayout = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const $rootPanelApi = useState(() => atom<GridviewApi | null>(null))[0];
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      $rootPanelApi.set(event.api);
      initializeRootPanelLayout(event.api);
    },
    [$rootPanelApi]
  );
  useResizeMainPanelOnFirstVisit($rootPanelApi, ref);

  return (
    <AutoLayoutProvider $rootApi={$rootPanelApi}>
      <GridviewReact
        ref={ref}
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
UpscalingTabAutoLayout.displayName = 'UpscalingTabAutoLayout';
