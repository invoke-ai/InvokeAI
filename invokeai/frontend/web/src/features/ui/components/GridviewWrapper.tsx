import 'dockview/dist/styles/dockview.css';
import './dockview_theme_invoke.css';

import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { GridviewApi, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { BoardsListPanelContent } from 'features/gallery/components/BoardsListPanelContent';
import { Gallery } from 'features/gallery/components/Gallery';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import ProgressImage from 'features/gallery/components/ImageViewer/ProgressImage';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import QueueControls from 'features/queue/components/QueueControls';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/components/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { ImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { MainPanelContent } from 'features/ui/components/MainPanelContent';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { atom } from 'nanostores';
import { memo, useCallback, useEffect } from 'react';

const LayersPanelContent = memo(() => (
  <CanvasManagerProviderGate>
    <CanvasLayersPanelContent />
  </CanvasManagerProviderGate>
));
LayersPanelContent.displayName = 'LayersPanelContent';

const ViewerPanelContent = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
    <ViewerToolbar />
    <Divider />
    <ImageViewer />
  </Flex>
));
ViewerPanelContent.displayName = 'ViewerPanelContent';

const ProgressPanelContent = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2}>
    <ProgressImage />
  </Flex>
));
ProgressPanelContent.displayName = 'ProgressPanelContent';

const components: IGridviewReactProps['components'] = {
  main: MainPanelContent,
  boards: BoardsListPanelContent,
  gallery: Gallery,
  layers: LayersPanelContent,
  queueControls: QueueControls,
  prompts: Prompts,
  imageSettings: ImageSettingsAccordion,
  generationSettings: GenerationSettingsAccordion,
  compositingSettings: CompositingSettingsAccordion,
  advancedSettings: AdvancedSettingsAccordion,
  refinerSettings: RefinerSettingsAccordion,
  viewer: ViewerPanelContent,
  progress: ProgressPanelContent,
};

export const $panels = atom<{ api: GridviewApi; resetLayout: () => void } | null>(null);

const canvasLayout = (api: GridviewApi) => {
  api.clear();
  const mainPanel = api.addPanel({
    id: 'main',
    component: 'main',
    minimumWidth: 200,
    priority: LayoutPriority.High,
  });
  // api.addPanel({
  //   id: 'viewer',
  //   component: 'viewer',
  //   position: {
  //     direction: 'within',
  //     referencePanel: mainPanel.id,
  //   },
  // });
  // api.addPanel({
  //   id: 'progress',
  //   component: 'progress',
  //   position: {
  //     direction: 'within',
  //     referencePanel: mainPanel.id,
  //   },
  // });
  const queueControls = api.addPanel({
    id: 'queue-controls',
    component: 'queueControls',
    // floating: true,
    // initialHeight: 48 + 24,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: mainPanel.id,
    },
    priority: LayoutPriority.Low,
    snap: true,
  });
  const promptsPanel = api.addPanel({
    id: 'prompts',
    component: 'prompts',
    position: {
      direction: 'below',
      referencePanel: queueControls.id,
    },
  });
  const imagePanel = api.addPanel({
    id: 'imageSettings',
    component: 'imageSettings',
    position: {
      direction: 'below',
      referencePanel: promptsPanel.id,
    },
  });
  const genPanel = api.addPanel({
    id: 'generationSettings',
    component: 'generationSettings',
    position: {
      direction: 'below',
      referencePanel: imagePanel.id,
    },
  });
  const compPanel = api.addPanel({
    id: 'compositingSettings',
    component: 'compositingSettings',
    position: {
      direction: 'below',
      referencePanel: genPanel.id,
    },
  });
  const advancedPanel = api.addPanel({
    id: 'advancedSettings',
    component: 'advancedSettings',
    position: {
      direction: 'below',
      referencePanel: compPanel.id,
    },
  });
  api.addPanel({
    id: 'refinerSettings',
    component: 'refinerSettings',
    position: {
      direction: 'below',
      referencePanel: advancedPanel.id,
    },
  });

  const galleryPanel = api.addPanel({
    id: 'gallery',
    component: 'gallery',
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: mainPanel.id,
    },
    snap: true,
  });
  const boardsPanel = api.addPanel({
    id: 'boards',
    component: 'boards',
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'above',
      referencePanel: galleryPanel.id,
    },
    priority: LayoutPriority.Low,
  });
  mainPanel.api.setActive();
  galleryPanel.api.setSize({ width: RIGHT_PANEL_MIN_SIZE_PX });
  boardsPanel.api.setSize({ height: 200 });
  queueControls.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
};

const galleryLayout = (api: GridviewApi) => {
  api.clear();
  const viewer = api.addPanel({
    id: 'gallery-viewer',
    component: 'viewer',
  });
  const gallery = api.addPanel({
    id: 'gallery-gallery',
    component: 'gallery',
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: viewer.id,
    },
  });
  api.addPanel({
    id: 'gallery-boards',
    component: 'boards',
    position: {
      direction: 'above',
      referencePanel: gallery.id,
    },
  });
  viewer.api.setActive();
};

export const GridviewWrapper = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  const onTabChange = useCallback((tab: TabName, api: GridviewApi) => {
    if (tab === 'gallery') {
      galleryLayout(api);
    } else {
      canvasLayout(api);
    }
  }, []);

  const onReady = useCallback<IGridviewReactProps['onReady']>((event) => {
    const _resetToDefaults = () => canvasLayout(event.api);
    $panels.set({ api: event.api, resetLayout: _resetToDefaults });
    _resetToDefaults();
  }, []);

  useEffect(() => {
    const panels = $panels.get();
    if (!panels) {
      return;
    }
    onTabChange(tab, panels.api);
  }, [onTabChange, tab]);

  return (
    <GridviewReact
      className="dockview-theme-invoke"
      components={components}
      onReady={onReady}
      orientation={Orientation.VERTICAL}
    />
  );
});
GridviewWrapper.displayName = 'GridviewWrapper';
