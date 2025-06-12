import 'dockview/dist/styles/dockview.css';
import './dockview_theme_invoke.css';

import { Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  type DockviewApi,
  DockviewDefaultTab,
  DockviewReact,
  type DockviewTheme,
  type IDockviewHeaderActionsProps,
  type IDockviewPanelHeaderProps,
  type IDockviewReactProps,
} from 'dockview';
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
import { memo, useCallback, useEffect, useState } from 'react';
import { PiArrowSquareOutBold, PiCornersInBold, PiCornersOutBold } from 'react-icons/pi';

const MyCustomTab = (props: IDockviewPanelHeaderProps) => {
  const onDragEnter = useCallback(() => {
    if (!props.api.isActive) {
      props.api.setActive();
    }
  }, [props.api]);

  return <DockviewDefaultTab hideClose {...props} onDragEnter={onDragEnter} />;
};

const RightHeaderActions = (props: IDockviewHeaderActionsProps) => {
  const [isMaximized, setIsMaximized] = useState(false);
  const popOutToFloating = useCallback(() => {
    props.containerApi.addFloatingGroup(props.group);
  }, [props.containerApi, props.group]);
  const maximize = useCallback(() => {
    if (!props.group.activePanel) {
      props.group.panels.at(0)?.api.setActive();
    }
    if (!props.group.activePanel) {
      return;
    }
    props.containerApi.maximizeGroup(props.group.activePanel);
  }, [props.containerApi, props.group]);
  const exitMaximized = useCallback(() => {
    if (!props.group.activePanel) {
      props.group.panels.at(0)?.api.setActive();
    }
    if (!props.group.activePanel) {
      return;
    }
    props.containerApi.exitMaximizedGroup();
  }, [props.containerApi, props.group]);

  useEffect(() => {
    const subscription = props.containerApi.onDidMaximizedGroupChange((e) => {
      if (e.group.id === props.group.id) {
        setIsMaximized(e.isMaximized);
      }
    });

    return () => {
      subscription.dispose();
    };
  }, [props.containerApi, props.group.id]);

  return (
    <Flex h="full" alignItems="center" pe={1}>
      {!isMaximized && (
        <IconButton
          size="xs"
          variant="link"
          alignSelf="stretch"
          icon={<PiCornersOutBold />}
          aria-label="Maximize Panel"
          tooltip="Maximize Panel"
          onClick={maximize}
          opacity={0.7}
        />
      )}
      {isMaximized && (
        <IconButton
          size="xs"
          variant="link"
          alignSelf="stretch"
          icon={<PiCornersInBold />}
          aria-label="Maximize Panel"
          tooltip="Maximize Panel"
          onClick={exitMaximized}
          opacity={0.7}
        />
      )}
      {props.group.api.location.type !== 'floating' && (
        <IconButton
          size="xs"
          variant="link"
          alignSelf="stretch"
          icon={<PiArrowSquareOutBold />}
          aria-label="Pop out Panel"
          tooltip="Pop out Panel"
          onClick={popOutToFloating}
          opacity={0.7}
        />
      )}
    </Flex>
  );
};

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

const components: IDockviewReactProps['components'] = {
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

const theme: DockviewTheme = {
  className: 'dockview-theme-invoke',
  name: 'Invoke',
};

export const $panels = atom<{ api: DockviewApi; resetLayout: () => void } | null>(null);

const canvasLayout = (api: DockviewApi) => {
  api.clear();
  const mainPanel = api.addPanel({
    id: 'main',
    component: 'main',
    title: 'Workspace',
    minimumWidth: 200,
  });
  api.addPanel({
    id: 'viewer',
    component: 'viewer',
    title: 'Viewer',
    position: {
      direction: 'within',
      referencePanel: mainPanel,
    },
  });
  api.addPanel({
    id: 'progress',
    component: 'progress',
    title: 'Progress',
    position: {
      direction: 'within',
      referencePanel: mainPanel,
    },
  });
  const queueControls = api.addPanel({
    id: 'queue-controls',
    title: 'Queue',
    component: 'queueControls',
    // floating: true,
    // initialHeight: 48 + 24,
    initialHeight: 48,
    maximumHeight: 48,
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    initialWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: mainPanel,
    },
  });
  const promptsPanel = api.addPanel({
    id: 'prompts',
    title: 'Prompts',
    component: 'prompts',
    position: {
      direction: 'below',
      referencePanel: queueControls,
    },
  });
  const imagePanel = api.addPanel({
    id: 'imageSettings',
    title: 'Image',
    component: 'imageSettings',
    position: {
      direction: 'below',
      referencePanel: promptsPanel,
    },
  });
  api.addPanel({
    id: 'generationSettings',
    title: 'Generation',
    component: 'generationSettings',
    position: {
      direction: 'within',
      referencePanel: imagePanel,
    },
  });
  const compPanel = api.addPanel({
    id: 'compositingSettings',
    title: 'Compositing',
    component: 'compositingSettings',
    position: {
      direction: 'below',
      referencePanel: imagePanel,
    },
  });
  const advancedPanel = api.addPanel({
    id: 'advancedSettings',
    title: 'Advanced',
    component: 'advancedSettings',
    position: {
      direction: 'within',
      referencePanel: compPanel,
    },
  });
  api.addPanel({
    id: 'refinerSettings',
    title: 'Refiner',
    component: 'refinerSettings',
    position: {
      direction: 'within',
      referencePanel: advancedPanel,
    },
  });
  const boardsPanel = api.addPanel({
    id: 'boards',
    component: 'boards',
    title: 'Boards',
    initialWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: mainPanel,
    },
  });
  const galleryPanel = api.addPanel({
    id: 'gallery',
    component: 'gallery',
    title: 'Gallery',
    position: {
      direction: 'below',
      referencePanel: boardsPanel,
    },
  });
  api.addPanel({
    id: 'layers',
    component: 'layers',
    title: 'Layers',
    position: {
      direction: 'below',
      referencePanel: galleryPanel,
    },
  });
  mainPanel.api.setActive();
};

const galleryLayout = (api: DockviewApi) => {
  api.clear();
  const viewer = api.addPanel({
    id: 'gallery-viewer',
    title: 'Viewer',
    component: 'viewer',
  });
  api.addPanel({
    id: 'gallery-progress',
    title: 'Progress',
    component: 'progress',
    position: {
      direction: 'within',
      referencePanel: viewer,
    },
  });
  const gallery = api.addPanel({
    id: 'gallery-gallery',
    title: 'Gallery',
    component: 'gallery',
    initialWidth: RIGHT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'right',
      referencePanel: viewer,
    },
  });
  api.addPanel({
    id: 'gallery-boards',
    title: 'Boards',
    component: 'boards',
    position: {
      direction: 'above',
      referencePanel: gallery,
    },
  });
  viewer.api.setActive();
};

export const DockviewWrapper = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  const onTabChange = useCallback((tab: TabName, api: DockviewApi) => {
    if (tab === 'gallery') {
      galleryLayout(api);
    } else {
      canvasLayout(api);
    }
  }, []);

  const onReady = useCallback<IDockviewReactProps['onReady']>(
    (event) => {
      const _resetToDefaults = () => canvasLayout(event.api);
      $panels.set({ api: event.api, resetLayout: _resetToDefaults });
      onTabChange(tab, event.api);
    },
    [onTabChange, tab]
  );

  useEffect(() => {
    const panels = $panels.get();
    if (!panels) {
      return;
    }
    onTabChange(tab, panels.api);
  }, [onTabChange, tab]);

  return (
    <DockviewReact
      components={components}
      onReady={onReady}
      theme={theme}
      defaultTabComponent={MyCustomTab}
      rightHeaderActionsComponent={RightHeaderActions}
    />
  );
});
DockviewWrapper.displayName = 'DockviewWrapper';
