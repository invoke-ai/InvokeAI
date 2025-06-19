import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, Orientation } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/CanvasLaunchpadPanel';
import { GenerateLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/GenerateLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { CanvasWorkspacePanel } from 'features/ui/layouts/CanvasWorkspacePanel';
import { GenerateLeftPanel } from 'features/ui/layouts/generate-tab-auto-layout';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { atom } from 'nanostores';
import { memo, useCallback, useEffect, useState } from 'react';

export const dockviewComponents: IDockviewReactProps['components'] = {
  // Shared components
  ImageViewer: ImageViewerPanel,
  GenerationProgress: GenerationProgressPanel,
  // Generate tab
  GenerateLaunchpad: GenerateLaunchpadPanel,
  // Upscaling tab
  UpscalingLaunchpad: GenerateLaunchpadPanel,
  // Workflows tab
  WorkflowsLaunchpad: GenerateLaunchpadPanel,
  // Canvas tab
  CanvasLaunchpad: CanvasLaunchpadPanel,
  CanvasWorkspace: CanvasWorkspacePanel,
};

const getGenerateLaunchpadPanel = (api: DockviewApi) => {
  return (
    api.getPanel('GenerateLaunchpad') ||
    api.addPanel({
      id: 'GenerateLaunchpad',
      component: 'GenerateLaunchpad',
      title: 'Launchpad',
    })
  );
};

const getCanvasLaunchpadPanel = (api: DockviewApi) => {
  return (
    api.getPanel('CanvasLaunchpad') ||
    api.addPanel({
      id: 'CanvasLaunchpad',
      component: 'CanvasLaunchpad',
      title: 'Launchpad',
    })
  );
};

const getCanvasWorkspacePanel = (api: DockviewApi) => {
  return (
    api.getPanel('CanvasWorkspace') ||
    api.addPanel({
      id: 'CanvasWorkspace',
      component: 'CanvasWorkspace',
      title: 'Canvas',
    })
  );
};

const getImageViewerPanel = (api: DockviewApi) => {
  return (
    api.getPanel('ImageViewer') ||
    api.addPanel({
      id: 'ImageViewer',
      component: 'ImageViewer',
      title: 'Image Viewer',
    })
  );
};

const getGenerationProgressPanel = (api: DockviewApi) => {
  return (
    api.getPanel('GenerationProgress') ||
    api.addPanel({
      id: 'GenerationProgress',
      component: 'GenerationProgress',
      title: 'Generation Progress',
    })
  );
};

const syncMainPanelLayout = (tab: TabName, api: DockviewApi) => {
  if (tab === 'generate') {
    const GenerateLaunchpad = getGenerateLaunchpadPanel(api);
    const ImageViewer = getImageViewerPanel(api);
    const GenerationProgress = getGenerationProgressPanel(api);
    const panelsToKeep = [GenerateLaunchpad.id, ImageViewer.id, GenerationProgress.id];
    for (const panel of api.panels) {
      if (!panelsToKeep.includes(panel.id)) {
        api.removePanel(panel);
      }
    }
  } else if (tab === 'canvas') {
    const CanvasLaunchpad = getCanvasLaunchpadPanel(api);
    const CanvasWorkspace = getCanvasWorkspacePanel(api);
    const ImageViewer = getImageViewerPanel(api);
    const GenerationProgress = getGenerationProgressPanel(api);
    const panelsToKeep = [CanvasLaunchpad.id, CanvasWorkspace.id, ImageViewer.id, GenerationProgress.id];
    for (const panel of api.panels) {
      if (!panelsToKeep.includes(panel.id)) {
        api.removePanel(panel);
      }
    }
  }
};

const MainPanel = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const [api, setApi] = useState<DockviewApi | null>(null);
  const onReady = useCallback<IDockviewReactProps['onReady']>((event) => {
    console.log('MainPanel onReady', event.api);
    setApi(event.api);
  }, []);

  useEffect(() => {
    if (api) {
      syncMainPanelLayout(tab, api);
    }
  }, [api, tab]);

  return (
    <Flex w="full" h="full">
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        defaultTabComponent={TabWithoutCloseButton}
        components={dockviewComponents}
        onReady={onReady}
        theme={dockviewTheme}
      />
    </Flex>
  );
});
MainPanel.displayName = 'MainPanel';

export const gridviewComponents: IGridviewReactProps['components'] = {
  // Shared components
  Gallery: GalleryPanel,
  Boards: BoardsPanel,
  Main: MainPanel,
  GenerateLeft: GenerateLeftPanel,
  CanvasLeft: GenerateLeftPanel,
  CanvasLayers: CanvasLayersPanel,
};

const syncGridviewLayout = (tab: TabName, api: GridviewApi) => {
  if (tab === 'generate') {
    const MainPanel =
      api.getPanel('Main') ??
      api.addPanel({
        id: 'Main',
        component: 'Main',
      });

    const GenerateLeftPanel =
      api.getPanel('GenerateLeft') ??
      api.addPanel({
        id: 'GenerateLeft',
        component: 'GenerateLeft',
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
        position: {
          direction: 'left',
          referencePanel: MainPanel.id,
        },
      });

    const GalleryPanel =
      api.getPanel('Gallery') ??
      api.addPanel({
        id: 'Gallery',
        component: 'Gallery',
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
        minimumHeight: 232,
        position: {
          direction: 'right',
          referencePanel: MainPanel.id,
        },
      });

    const BoardsPanel =
      api.getPanel('Boards') ??
      api.addPanel({
        id: 'Boards',
        component: 'Boards',
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
        minimumHeight: 36,
        position: {
          direction: 'above',
          referencePanel: GalleryPanel.id,
        },
      });

    const panelsToKeep = [MainPanel.id, GenerateLeftPanel.id, GalleryPanel.id, BoardsPanel.id];
    for (const panel of api.panels) {
      if (!panelsToKeep.includes(panel.id)) {
        api.removePanel(panel);
      }
    }
  } else if (tab === 'canvas') {
    const MainPanel =
      api.getPanel('Main') ??
      api.addPanel({
        id: 'Main',
        component: 'Main',
      });

    const CanvasLeftPanel =
      api.getPanel('CanvasLeft') ??
      api.addPanel({
        id: 'CanvasLeft',
        component: 'CanvasLeft',
        minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
        position: {
          direction: 'left',
          referencePanel: MainPanel.id,
        },
      });

    const GalleryPanel =
      api.getPanel('Gallery') ??
      api.addPanel({
        id: 'Gallery',
        component: 'Gallery',
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
        minimumHeight: 232,
        position: {
          direction: 'right',
          referencePanel: MainPanel.id,
        },
      });

    const BoardsPanel =
      api.getPanel('Boards') ??
      api.addPanel({
        id: 'Boards',
        component: 'Boards',
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
        minimumHeight: 36,
        position: {
          direction: 'above',
          referencePanel: GalleryPanel.id,
        },
      });

    const CanvasLayersPanel =
      api.getPanel('CanvasLayers') ??
      api.addPanel({
        id: 'CanvasLayers',
        component: 'CanvasLayers',
        minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
        minimumHeight: 232,
        position: {
          direction: 'below',
          referencePanel: GalleryPanel.id,
        },
      });

    const panelsToKeep = [MainPanel.id, CanvasLeftPanel.id, GalleryPanel.id, BoardsPanel.id, CanvasLayersPanel.id];
    for (const panel of api.panels) {
      if (!panelsToKeep.includes(panel.id)) {
        api.removePanel(panel);
      }
    }
  }
};

export const AutoLayout = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const $api = useState(() => atom<GridviewApi | null>(null))[0];
  const onReady = useCallback<IGridviewReactProps['onReady']>(
    (event) => {
      $api.set(event.api);
    },
    [$api]
  );
  useEffect(() => {
    const api = $api.get();
    if (api) {
      syncGridviewLayout(tab, api);
    }
  }, [$api, tab]);
  return (
    <AutoLayoutProvider $api={$api}>
      <GridviewReact
        className="dockview-theme-invoke"
        components={gridviewComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
AutoLayout.displayName = 'AutoLayout';
