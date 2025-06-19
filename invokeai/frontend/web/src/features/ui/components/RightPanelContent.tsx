import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GalleryTopBar } from 'features/gallery/components/GalleryTopBar';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { HorizontalResizeHandle } from 'features/ui/components/tabs/ResizeHandle';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useMemo, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

const FOCUS_REGION_STYLES: SystemStyleObject = {
  width: 'full',
  height: 'full',
  position: 'relative',
  flexDirection: 'column',
  display: 'flex',
};

export const RightPanelContent = memo(() => {
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: !!boardSearchText.length });
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const tab = useAppSelector(selectActiveTab);

  const boardsListPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'boards-list-panel',
      minSizePx: 128,
      defaultSizePx: 256,
      imperativePanelGroupRef,
      panelGroupDirection: 'vertical',
    }),
    []
  );
  const boardsListPanel = usePanel(boardsListPanelOptions);

  const galleryPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'gallery-panel',
      minSizePx: 128,
      defaultSizePx: 256,
      imperativePanelGroupRef,
      panelGroupDirection: 'vertical',
    }),
    []
  );
  const galleryPanel = usePanel(galleryPanelOptions);

  const canvasLayersPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'canvas-layers-panel',
      minSizePx: 128,
      defaultSizePx: 256,
      imperativePanelGroupRef,
      panelGroupDirection: 'vertical',
    }),
    []
  );
  const canvasLayersPanel = usePanel(canvasLayersPanelOptions);

  return (
    <FocusRegionWrapper region="gallery" sx={FOCUS_REGION_STYLES}>
      <GalleryTopBar boardsListPanel={boardsListPanel} boardSearchDisclosure={boardSearchDisclosure} />
      <PanelGroup ref={imperativePanelGroupRef} direction="vertical" autoSaveId="boards-list-panel">
        <Panel order={0} id="boards-panel" collapsible {...boardsListPanel.panelProps}>
          <BoardsPanel boardSearchDisclosure={boardSearchDisclosure} />
        </Panel>
        <HorizontalResizeHandle id="boards-list-to-gallery-panel-handle" {...boardsListPanel.resizeHandleProps} />
        <Panel order={1} id="gallery-wrapper-panel" collapsible {...galleryPanel.panelProps}>
          <GalleryPanel />
        </Panel>
        {tab === 'canvas' && (
          <>
            <HorizontalResizeHandle id="gallery-panel-to-layers-handle" {...galleryPanel.resizeHandleProps} />
            <Panel order={2} id="canvas-layers-panel" collapsible {...canvasLayersPanel.panelProps}>
              <CanvasManagerProviderGate>
                <CanvasLayersPanel />
              </CanvasManagerProviderGate>
            </Panel>
          </>
        )}
      </PanelGroup>
    </FocusRegionWrapper>
  );
});
RightPanelContent.displayName = 'RightPanelContent';
