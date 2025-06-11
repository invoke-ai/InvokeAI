import 'dockview/dist/styles/dockview.css';
import './dockview_theme_invoke.css';

import { Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type {
  DockviewApi,
  DockviewTheme,
  IDockviewHeaderActionsProps,
  IDockviewPanelHeaderProps,
  IDockviewReactProps,
} from 'dockview';
import { DockviewDefaultTab, DockviewReact } from 'dockview';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
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
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { MainPanelContent } from 'features/ui/components/MainPanelContent';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import {
  $isLeftPanelOpen,
  $isRightPanelOpen,
  LEFT_PANEL_MIN_SIZE_PX,
  RIGHT_PANEL_MIN_SIZE_PX,
  selectWithLeftPanel,
  selectWithRightPanel,
} from 'features/ui/store/uiSlice';
import { atom } from 'nanostores';
import type { CSSProperties } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiArrowSquareOutBold, PiCornersInBold, PiCornersOutBold } from 'react-icons/pi';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%', minWidth: 0 };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

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
          size="sm"
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
          size="sm"
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
          size="sm"
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

const components: IDockviewReactProps['components'] = {
  main: MainPanelContent,
  boards: BoardsListPanelContent,
  gallery: Gallery,
  layers: () => (
    <CanvasManagerProviderGate>
      <CanvasLayersPanelContent />
    </CanvasManagerProviderGate>
  ),
  queueControls: QueueControls,
  prompts: Prompts,
  imageSettings: ImageSettingsAccordion,
  generationSettings: GenerationSettingsAccordion,
  compositingSettings: CompositingSettingsAccordion,
  advancedSettings: AdvancedSettingsAccordion,
  refinerSettings: RefinerSettingsAccordion,
  viewer: () => (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
      <ViewerToolbar />
      <Divider />
      <ImageViewer />
    </Flex>
  ),
  progress: () => (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2}>
      <ProgressImage />
    </Flex>
  ),
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

export const AppContent = memo(() => {
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const tab = useAppSelector(selectActiveTab);
  useDndMonitor();

  const withLeftPanel = useAppSelector(selectWithLeftPanel);
  const leftPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'left-panel',
      minSizePx: LEFT_PANEL_MIN_SIZE_PX,
      defaultSizePx: LEFT_PANEL_MIN_SIZE_PX,
      imperativePanelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onLeftPanelCollapse,
    }),
    []
  );
  const leftPanel = usePanel(leftPanelUsePanelOptions);
  useRegisteredHotkeys({
    id: 'toggleLeftPanel',
    category: 'app',
    callback: leftPanel.toggle,
    options: { enabled: withLeftPanel },
    dependencies: [leftPanel.toggle, withLeftPanel],
  });

  const withRightPanel = useAppSelector(selectWithRightPanel);
  const rightPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'right-panel',
      minSizePx: RIGHT_PANEL_MIN_SIZE_PX,
      defaultSizePx: RIGHT_PANEL_MIN_SIZE_PX,
      imperativePanelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onRightPanelCollapse,
    }),
    []
  );
  const rightPanel = usePanel(rightPanelUsePanelOptions);
  useRegisteredHotkeys({
    id: 'toggleRightPanel',
    category: 'app',
    callback: rightPanel.toggle,
    options: { enabled: withRightPanel },
    dependencies: [rightPanel.toggle, withRightPanel],
  });

  useRegisteredHotkeys({
    id: 'resetPanelLayout',
    category: 'app',
    callback: () => {
      leftPanel.reset();
      rightPanel.reset();
    },
    dependencies: [leftPanel.reset, rightPanel.reset],
  });
  useRegisteredHotkeys({
    id: 'togglePanels',
    category: 'app',
    callback: () => {
      if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
        leftPanel.expand();
        rightPanel.expand();
      } else {
        leftPanel.collapse();
        rightPanel.collapse();
      }
    },
    dependencies: [
      leftPanel.isCollapsed,
      rightPanel.isCollapsed,
      leftPanel.expand,
      rightPanel.expand,
      leftPanel.collapse,
      rightPanel.collapse,
    ],
  });

  const onReady = useCallback<IDockviewReactProps['onReady']>((event) => {
    const _resetToDefaults = () => canvasLayout(event.api);
    $panels.set({ api: event.api, resetLayout: _resetToDefaults });
    _resetToDefaults();
  }, []);

  useEffect(() => {
    const panels = $panels.get();
    if (!panels) {
      return;
    }
    if (tab === 'gallery') {
      galleryLayout(panels.api);
    } else {
      canvasLayout(panels.api);
    }
  }, [tab]);

  return (
    <Flex id="invoke-app-tabs" w="full" h="full" overflow="hidden">
      <VerticalNavBar />
      <DockviewReact
        components={components}
        onReady={onReady}
        theme={theme}
        defaultTabComponent={MyCustomTab}
        rightHeaderActionsComponent={RightHeaderActions}
      />
      {/* <PanelGroup
        ref={imperativePanelGroupRef}
        id="app-panel-group"
        autoSaveId="app-panel-group"
        direction="horizontal"
        style={panelStyles}
      >
        {withLeftPanel && (
          <>
            <Panel id="left-panel" order={0} collapsible style={panelStyles} {...leftPanel.panelProps}>
              <LeftPanelContent />
            </Panel>
            <VerticalResizeHandle id="left-main-handle" {...leftPanel.resizeHandleProps} />
          </>
        )}
        <Panel id="main-panel" order={1} minSize={20} style={panelStyles}>
          <MainPanelContent />
          {withLeftPanel && <FloatingLeftPanelButtons onToggle={leftPanel.toggle} />}
          {withRightPanel && <FloatingRightPanelButtons onToggle={rightPanel.toggle} />}
        </Panel>
        {withRightPanel && (
          <>
            <VerticalResizeHandle id="main-right-handle" {...rightPanel.resizeHandleProps} />
            <Panel id="right-panel" order={2} style={panelStyles} collapsible {...rightPanel.panelProps}>
              <RightPanelContent />
            </Panel>
          </>
        )}
      </PanelGroup> */}
    </Flex>
  );
});
AppContent.displayName = 'AppContent';

const GalleryTab = () => {
  const onReady = useCallback<IDockviewReactProps['onReady']>((event) => {
    const viewer = event.api.addPanel({
      id: 'gallery-viewer',
      title: 'Viewer',
      component: 'viewer',
    });
    const gallery = event.api.addPanel({
      id: 'gallery-gallery',
      title: 'Gallery',
      component: 'gallery',
      initialWidth: RIGHT_PANEL_MIN_SIZE_PX,
      position: {
        direction: 'right',
        referencePanel: viewer,
      },
    });
    event.api.addPanel({
      id: 'gallery-boards',
      title: 'Boards',
      component: 'boards',
      position: {
        direction: 'above',
        referencePanel: gallery,
      },
    });
  }, []);
  return (
    <DockviewReact
      onReady={onReady}
      components={components}
      theme={theme}
      defaultTabComponent={MyCustomTab}
      rightHeaderActionsComponent={RightHeaderActions}
    />
  );
};
