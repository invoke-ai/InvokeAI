import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasEditor } from 'features/controlLayers/components/ControlLayersEditor';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { useIsImageViewerOpen } from 'features/gallery/components/ImageViewer/useImageViewer';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import { TabMountGate } from 'features/ui/components/TabMountGate';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import NodesTab from 'features/ui/components/tabs/NodesTab';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import { TabVisibilityGate } from 'features/ui/components/TabVisibilityGate';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import { $isGalleryPanelOpen, $isParametersPanelOpen } from 'features/ui/store/uiSlice';
import type { TabName } from "features/ui/store/uiTypes";
import type { CSSProperties } from 'react';
import { memo, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import ParametersPanelUpscale from './ParametersPanels/ParametersPanelUpscale';
import ResizeHandle from './tabs/ResizeHandle';

const TABS_WITH_GALLERY_PANEL: TabName[] = ['generation', 'upscaling', 'workflows'] as const;
const TABS_WITH_OPTIONS_PANEL: TabName[] = ['generation', 'upscaling', 'workflows'] as const;

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%' };
const GALLERY_MIN_SIZE_PX = 310;
const GALLERY_MIN_SIZE_PCT = 20;
const OPTIONS_PANEL_MIN_SIZE_PX = 430;
const OPTIONS_PANEL_MIN_SIZE_PCT = 20;

const onGalleryPanelCollapse = (isCollapsed: boolean) => $isGalleryPanelOpen.set(!isCollapsed);
const onParametersPanelCollapse = (isCollapsed: boolean) => $isParametersPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const isImageViewerOpen = useIsImageViewerOpen();
  const shouldShowGalleryPanel = useAppSelector((s) => TABS_WITH_GALLERY_PANEL.includes(s.ui.activeTab));
  const shouldShowOptionsPanel = useAppSelector((s) => TABS_WITH_OPTIONS_PANEL.includes(s.ui.activeTab));
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('gallery', ref);

  const optionsPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'options-panel',
      unit: 'pixels',
      minSize: OPTIONS_PANEL_MIN_SIZE_PX,
      defaultSize: OPTIONS_PANEL_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onParametersPanelCollapse,
    }),
    []
  );

  const galleryPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'gallery-panel',
      unit: 'pixels',
      minSize: GALLERY_MIN_SIZE_PX,
      defaultSize: GALLERY_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onGalleryPanelCollapse,
    }),
    []
  );

  const panelStorage = usePanelStorage();

  const optionsPanel = usePanel(optionsPanelUsePanelOptions);

  const galleryPanel = usePanel(galleryPanelUsePanelOptions);

  useHotkeys('g', galleryPanel.toggle, [galleryPanel.toggle]);
  useHotkeys(['t', 'o'], optionsPanel.toggle, [optionsPanel.toggle]);
  useHotkeys(
    'shift+r',
    () => {
      optionsPanel.reset();
      galleryPanel.reset();
    },
    [optionsPanel.reset, galleryPanel.reset]
  );
  useHotkeys(
    'f',
    () => {
      if (optionsPanel.isCollapsed || galleryPanel.isCollapsed) {
        optionsPanel.expand();
        galleryPanel.expand();
      } else {
        optionsPanel.collapse();
        galleryPanel.collapse();
      }
    },
    [
      optionsPanel.isCollapsed,
      galleryPanel.isCollapsed,
      optionsPanel.expand,
      galleryPanel.expand,
      optionsPanel.collapse,
      galleryPanel.collapse,
    ]
  );

  return (
    <Flex ref={ref} id="invoke-app-tabs" w="full" h="full" gap={4} p={4}>
      <VerticalNavBar />
      <Flex position="relative" w="full" h="full" gap={4}>
        <PanelGroup
          ref={panelGroupRef}
          id="app-panel-group"
          autoSaveId="app"
          direction="horizontal"
          style={panelStyles}
          storage={panelStorage}
        >
          <Panel order={0} collapsible style={panelStyles} {...optionsPanel.panelProps}>
            <TabMountGate tab="generation">
              <TabVisibilityGate tab="generation">
                <ParametersPanelTextToImage />
              </TabVisibilityGate>
            </TabMountGate>
            <TabMountGate tab="upscaling">
              <TabVisibilityGate tab="upscaling">
                <ParametersPanelUpscale />
              </TabVisibilityGate>
            </TabMountGate>
            <TabMountGate tab="workflows">
              <TabVisibilityGate tab="workflows">
                <NodeEditorPanelGroup />
              </TabVisibilityGate>
            </TabMountGate>
          </Panel>
          <ResizeHandle id="options-main-handle" orientation="vertical" {...optionsPanel.resizeHandleProps} />
          <Panel id="main-panel" order={1} minSize={20} style={panelStyles}>
            <TabMountGate tab="generation">
              <TabVisibilityGate tab="generation">
                <CanvasEditor />
              </TabVisibilityGate>
            </TabMountGate>
            {/* upscaling tab has no content of its own - uses image viewer only */}
            <TabMountGate tab="workflows">
              <TabVisibilityGate tab="workflows">
                <NodesTab />
              </TabVisibilityGate>
            </TabMountGate>
            {isImageViewerOpen && <ImageViewer />}
          </Panel>
          <ResizeHandle id="main-gallery-handle" orientation="vertical" {...galleryPanel.resizeHandleProps} />
          <Panel order={2} style={panelStyles} collapsible {...galleryPanel.panelProps}>
            <GalleryPanelContent />
          </Panel>
        </PanelGroup>
        {shouldShowOptionsPanel && <FloatingParametersPanelButtons panelApi={optionsPanel} />}
        {shouldShowGalleryPanel && <FloatingGalleryButton panelApi={galleryPanel} />}
        <TabMountGate tab="models">
          <TabVisibilityGate tab="models">
            <ModelManagerTab />
          </TabVisibilityGate>
        </TabMountGate>
        <TabMountGate tab="models">
          <TabVisibilityGate tab="queue">
            <QueueTab />
          </TabVisibilityGate>
        </TabMountGate>
      </Flex>
    </Flex>
  );
});

AppContent.displayName = 'AppContent';
