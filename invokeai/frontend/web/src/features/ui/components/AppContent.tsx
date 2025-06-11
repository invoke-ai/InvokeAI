import 'dockview/dist/styles/dockview.css';
import './dockview_theme_invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { DockviewApi, DockviewTheme, IDockviewPanelHeaderProps, IDockviewReactProps } from 'dockview';
import { DockviewDefaultTab, DockviewReact } from 'dockview';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { BoardsListPanelContent } from 'features/gallery/components/BoardsListPanelContent';
import { Gallery } from 'features/gallery/components/Gallery';
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
import { memo, useCallback, useMemo, useRef } from 'react';
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
};

const theme: DockviewTheme = {
  className: 'dockview-theme-invoke',
  name: 'Invoke',
};

const $dockviewApi = atom<DockviewApi | null>(null);

export const AppContent = memo(() => {
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
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
    $dockviewApi.set(event.api);
    const mainPanel = event.api.addPanel({
      id: 'main',
      component: 'main',
      title: 'Workspace',
      minimumWidth: 200,
    });
    const queueControls = event.api.addPanel({
      id: 'queue-controls',
      title: 'Queue Controls',
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
    const promptsPanel = event.api.addPanel({
      id: 'prompts',
      title: 'Prompts',
      component: 'prompts',
      position: {
        direction: 'below',
        referencePanel: queueControls,
      },
    });
    const imagePanel = event.api.addPanel({
      id: 'imageSettings',
      title: 'Image Settings',
      component: 'imageSettings',
      position: {
        direction: 'below',
        referencePanel: promptsPanel,
      },
    });
    const genPanel = event.api.addPanel({
      id: 'generationSettings',
      title: 'Generation Settings',
      component: 'generationSettings',
      position: {
        direction: 'within',
        referencePanel: imagePanel,
      },
    });
    const compPanel = event.api.addPanel({
      id: 'compositingSettings',
      title: 'Compositing Settings',
      component: 'compositingSettings',
      position: {
        direction: 'below',
        referencePanel: imagePanel,
      },
    });
    const advancedPanel = event.api.addPanel({
      id: 'advancedSettings',
      title: 'Advanced Settings',
      component: 'advancedSettings',
      position: {
        direction: 'within',
        referencePanel: compPanel,
      },
    });
    event.api.addPanel({
      id: 'refinerSettings',
      title: 'Refiner Settings',
      component: 'refinerSettings',
      position: {
        direction: 'within',
        referencePanel: advancedPanel,
      },
    });
    const boardsPanel = event.api.addPanel({
      id: 'boards',
      component: 'boards',
      initialWidth: RIGHT_PANEL_MIN_SIZE_PX,
      position: {
        direction: 'right',
        referencePanel: mainPanel,
      },
    });
    const galleryPanel = event.api.addPanel({
      id: 'gallery',
      component: 'gallery',
      title: 'Gallery',
      position: {
        direction: 'below',
        referencePanel: boardsPanel,
      },
    });
    event.api.addPanel({
      id: 'layers',
      component: 'layers',
      title: 'Layers',
      position: {
        direction: 'below',
        referencePanel: galleryPanel,
      },
    });
  }, []);

  return (
    <Flex id="invoke-app-tabs" w="full" h="full" overflow="hidden">
      <VerticalNavBar />
      <DockviewReact components={components} onReady={onReady} theme={theme} defaultTabComponent={MyCustomTab} />
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
