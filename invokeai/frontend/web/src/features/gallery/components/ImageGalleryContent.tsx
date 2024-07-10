import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Collapse, Flex, IconButton, Spacer, Tab, TabList, Tabs, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import GalleryHeader from 'features/gallery/components/GalleryHeader';
import { galleryViewChanged } from 'features/gallery/store/gallerySlice';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold } from 'react-icons/pi';
import { Panel, PanelGroup } from 'react-resizable-panels';

import BoardsList from './Boards/BoardsList/BoardsList';
import BoardsSearch from './Boards/BoardsList/BoardsSearch';
import GallerySettingsPopover from './GallerySettingsPopover/GallerySettingsPopover';
import GalleryImageGrid from './ImageGrid/GalleryImageGrid';
import { GalleryPagination } from './ImageGrid/GalleryPagination';
import { GallerySearch } from './ImageGrid/GallerySearch';

const baseStyles: ChakraProps['sx'] = {
  fontWeight: 'semibold',
  fontSize: 'sm',
  color: 'base.300',
};

const selectedStyles: ChakraProps['sx'] = {
  borderColor: 'base.800',
  borderBottomColor: 'base.900',
  color: 'invokeBlue.300',
};

const ImageGalleryContent = () => {
  const { t } = useTranslation();
  const galleryView = useAppSelector((s) => s.gallery.galleryView);
  const searchTerm = useAppSelector((s) => s.gallery.searchTerm);
  const boardSearchText = useAppSelector((s) => s.gallery.boardSearchText);
  const dispatch = useAppDispatch();
  const searchDisclosure = useDisclosure({ defaultIsOpen: false });
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: false });

  const handleClickImages = useCallback(() => {
    dispatch(galleryViewChanged('images'));
  }, [dispatch]);

  const handleClickAssets = useCallback(() => {
    dispatch(galleryViewChanged('assets'));
  }, [dispatch]);

  const panelGroupRef = useRef(null);

  return (
    <Flex position="relative" flexDirection="column" h="full" w="full" pt={2}>
      <Flex alignItems="center" gap={2}>
        <GalleryHeader />
        <GallerySettingsPopover />
        <Box position="relative" h="full">
          <IconButton
            w="full"
            h="full"
            onClick={boardSearchDisclosure.onToggle}
            tooltip={`${t('gallery.displayBoardSearch')}`}
            aria-label={t('gallery.displayBoardSearch')}
            icon={<PiMagnifyingGlassBold />}
            variant="link"
          />
          {boardSearchText && (
            <Box
              position="absolute"
              w={2}
              h={2}
              bg="invokeBlue.300"
              borderRadius="full"
              insetBlockStart={2}
              insetInlineEnd={2}
            />
          )}
        </Box>
      </Flex>
      <PanelGroup ref={panelGroupRef} direction="vertical">
        <Panel>
          <Collapse in={boardSearchDisclosure.isOpen}>
            <BoardsSearch />
          </Collapse>
          <BoardsList />
        </Panel>
        <ResizeHandle orientation="horizontal" />
        <Panel>
          <Flex flexDirection="column" alignItems="center" justifyContent="space-between" h="full" w="full">
            <Tabs index={galleryView === 'images' ? 0 : 1} variant="enclosed" display="flex" flexDir="column" w="full">
              <TabList gap={2} fontSize="sm" borderColor="base.800">
                <Tab sx={baseStyles} _selected={selectedStyles} onClick={handleClickImages} data-testid="images-tab">
                  {t('parameters.images')}
                </Tab>
                <Tab sx={baseStyles} _selected={selectedStyles} onClick={handleClickAssets} data-testid="assets-tab">
                  {t('gallery.assets')}
                </Tab>
                <Spacer />
                <Box position="relative">
                  <IconButton
                    w="full"
                    h="full"
                    onClick={searchDisclosure.onToggle}
                    tooltip={`${t('gallery.displaySearch')}`}
                    aria-label={t('gallery.displaySearch')}
                    icon={<PiMagnifyingGlassBold />}
                    variant="link"
                  />
                  {searchTerm && (
                    <Box
                      position="absolute"
                      w={2}
                      h={2}
                      bg="invokeBlue.300"
                      borderRadius="full"
                      insetBlockStart={2}
                      insetInlineEnd={2}
                    />
                  )}
                </Box>
              </TabList>
            </Tabs>
            <Box w="full">
              <Collapse in={searchDisclosure.isOpen}>
                <Box w="full" pt={2}>
                  <GallerySearch />
                </Box>
              </Collapse>
            </Box>
            <GalleryImageGrid />
            <GalleryPagination />
          </Flex>
        </Panel>
      </PanelGroup>
    </Flex>
  );
};

export default memo(ImageGalleryContent);
