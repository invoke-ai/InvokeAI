import { Box, Collapse, Divider, Flex, IconButton, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { usePanel, type UsePanelOptions } from 'features/ui/hooks/usePanel';
import type { CSSProperties } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { MdSearch, MdSearchOff } from 'react-icons/md';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import BoardsListWrapper from './Boards/BoardsList/BoardsListWrapper';
import BoardsSearch from './Boards/BoardsList/BoardsSearch';
import { Gallery } from './Gallery';
import GallerySettingsPopover from './GallerySettingsPopover/GallerySettingsPopover';

export const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

const ImageGalleryContent = () => {
  const { t } = useTranslation();
  const boardSearchText = useAppSelector((s) => s.gallery.boardSearchText);
  const dispatch = useAppDispatch();
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: !!boardSearchText.length });
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const boardsListPanelOptions = useMemo<UsePanelOptions>(
    () => ({
      unit: 'pixels',
      minSize: 128,
      defaultSize: 256,
      fallbackMinSizePct: 20,
      panelGroupRef,
      panelGroupDirection: 'vertical',
    }),
    []
  );
  const boardsListPanel = usePanel(boardsListPanelOptions);

  const handleClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
      boardSearchDisclosure.onToggle();
    } else {
      boardSearchDisclosure.onToggle();
    }
  }, [boardSearchText, dispatch, boardSearchDisclosure]);

  useEffect(() => {
    if (boardSearchDisclosure.isOpen) {
      boardsListPanel.expand();
    }
  }, [boardSearchDisclosure, boardsListPanel]);

  return (
    <Flex position="relative" flexDirection="column" h="full" w="full" pt={2}>
      <Flex alignItems="center" gap={2}>
        <GalleryHeader onClickBoardName={boardsListPanel.toggle} />
        <GallerySettingsPopover />
        <Box position="relative" h="full">
          <IconButton
            w="full"
            h="full"
            onClick={handleClickBoardSearch}
            tooltip={
              boardSearchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`
            }
            aria-label={t('gallery.displayBoardSearch')}
            icon={boardSearchText.length ? <MdSearchOff /> : <MdSearch />}
            variant="link"
          />
        </Box>
      </Flex>

      <PanelGroup ref={panelGroupRef} direction="vertical">
        <Panel
          id="boards-list-panel"
          ref={boardsListPanel.ref}
          defaultSize={boardsListPanel.defaultSize}
          minSize={boardsListPanel.minSize}
          onCollapse={boardsListPanel.onCollapse}
          onExpand={boardsListPanel.onExpand}
          collapsible
        >
          <Flex flexDir="column" w="full" h="full">
            <Collapse in={boardSearchDisclosure.isOpen} style={COLLAPSE_STYLES}>
              <BoardsSearch />
            </Collapse>
            <Divider pt={2} />
            <BoardsListWrapper />
          </Flex>
        </Panel>
        <ResizeHandle
          id="gallery-panel-handle"
          orientation="horizontal"
          onDoubleClick={boardsListPanel.onDoubleClickHandle}
        />
        <Panel id="gallery-wrapper-panel" minSize={20}>
          <Gallery />
        </Panel>
      </PanelGroup>
    </Flex>
  );
};

export default memo(ImageGalleryContent);
