import {
  Box,
  Button,
  Collapse,
  Divider,
  Flex,
  IconButton,
  type SystemStyleObject,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSessionType } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { HorizontalResizeHandle } from 'features/ui/components/tabs/ResizeHandle';
import { usePanel, type UsePanelOptions } from 'features/ui/hooks/usePanel';
import type { CSSProperties } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCaretUpBold, PiMagnifyingGlassBold } from 'react-icons/pi';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import BoardsListWrapper from './Boards/BoardsList/BoardsListWrapper';
import BoardsSearch from './Boards/BoardsList/BoardsSearch';
import BoardsSettingsPopover from './Boards/BoardsSettingsPopover';
import { Gallery } from './Gallery';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

const FOCUS_REGION_STYLES: SystemStyleObject = {
  width: 'full',
  height: 'full',
  position: 'relative',
  flexDirection: 'column',
  display: 'flex',
};

const GalleryPanelContent = () => {
  const { t } = useTranslation();
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const dispatch = useAppDispatch();
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: !!boardSearchText.length });
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const sessionType = useAppSelector(selectCanvasSessionType);

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

  const handleClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
    }
    boardSearchDisclosure.onToggle();
    boardsListPanel.expand();
  }, [boardSearchText.length, boardSearchDisclosure, boardsListPanel, dispatch]);

  return (
    <FocusRegionWrapper region="gallery" sx={FOCUS_REGION_STYLES}>
      <Flex alignItems="center" justifyContent="space-between" w="full">
        <Flex flexGrow={1} flexBasis={0}>
          <Button
            size="sm"
            variant="ghost"
            onClick={boardsListPanel.toggle}
            rightIcon={boardsListPanel.isCollapsed ? <PiCaretDownBold /> : <PiCaretUpBold />}
          >
            {boardsListPanel.isCollapsed ? t('boards.viewBoards') : t('boards.hideBoards')}
          </Button>
        </Flex>
        <Flex>
          <GalleryHeader />
        </Flex>
        <Flex flexGrow={1} flexBasis={0} justifyContent="flex-end">
          <BoardsSettingsPopover />
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={handleClickBoardSearch}
            tooltip={
              boardSearchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`
            }
            aria-label={t('gallery.displayBoardSearch')}
            icon={<PiMagnifyingGlassBold />}
            colorScheme={boardSearchDisclosure.isOpen ? 'invokeBlue' : 'base'}
          />
        </Flex>
      </Flex>

      <PanelGroup ref={imperativePanelGroupRef} direction="vertical" autoSaveId="boards-list-panel">
        <Panel order={0} id="boards-panel" collapsible {...boardsListPanel.panelProps}>
          <Flex flexDir="column" w="full" h="full">
            <Collapse in={boardSearchDisclosure.isOpen} style={COLLAPSE_STYLES}>
              <Box w="full" pt={2}>
                <BoardsSearch />
              </Box>
            </Collapse>
            <Divider pt={2} />
            <BoardsListWrapper />
          </Flex>
        </Panel>
        <HorizontalResizeHandle id="boards-list-to-gallery-panel-handle" {...boardsListPanel.resizeHandleProps} />
        <Panel order={1} id="gallery-wrapper-panel" collapsible {...galleryPanel.panelProps}>
          <Gallery />
        </Panel>
        {sessionType === 'advanced' && (
          <>
            <HorizontalResizeHandle id="gallery-panel-to-layers-handle" {...galleryPanel.resizeHandleProps} />
            <Panel order={2} id="canvas-layers-panel" collapsible {...canvasLayersPanel.panelProps}>
              <CanvasManagerProviderGate>
                <CanvasLayersPanelContent />
              </CanvasManagerProviderGate>
            </Panel>
          </>
        )}
      </PanelGroup>
    </FocusRegionWrapper>
  );
};

export default memo(GalleryPanelContent);
