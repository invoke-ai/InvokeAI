import { Collapse, Flex, Icon, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGallerySlice, virtualBoardsSectionOpenChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCalendarBold, PiCaretDownBold, PiCaretRightBold } from 'react-icons/pi';
import { useListVirtualBoardsByDateQuery } from 'services/api/endpoints/virtual_boards';

import VirtualBoardItem from './VirtualBoardItem';

const selectShowVirtualBoards = createSelector(selectGallerySlice, (gallery) => gallery.showVirtualBoards);
const selectVirtualBoardsSectionOpen = createSelector(
  selectGallerySlice,
  (gallery) => gallery.virtualBoardsSectionOpen
);

export const VirtualBoardSection = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showVirtualBoards = useAppSelector(selectShowVirtualBoards);
  const isOpen = useAppSelector(selectVirtualBoardsSectionOpen);

  const { data: virtualBoards } = useListVirtualBoardsByDateQuery(undefined, {
    skip: !showVirtualBoards,
  });

  const toggleOpen = useCallback(() => {
    dispatch(virtualBoardsSectionOpenChanged(!isOpen));
  }, [dispatch, isOpen]);

  if (!showVirtualBoards || !virtualBoards?.length) {
    return null;
  }

  return (
    <Flex direction="column">
      <Flex w="full" justifyContent="space-between" alignItems="center" ps={2} py={1}>
        <Flex alignItems="center" gap={1}>
          <Icon as={PiCalendarBold} color="base.500" boxSize={4} />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {t('boards.byDate')}
          </Text>
        </Flex>
        <IconButton
          size="sm"
          variant="link"
          aria-label={isOpen ? t('common.collapse') : t('common.expand')}
          icon={isOpen ? <PiCaretDownBold /> : <PiCaretRightBold />}
          onClick={toggleOpen}
        />
      </Flex>
      <Collapse in={isOpen}>
        <Flex direction="column" gap={1}>
          {virtualBoards.map((board) => (
            <VirtualBoardItem key={board.virtual_board_id} board={board} />
          ))}
        </Flex>
      </Collapse>
    </Flex>
  );
});

VirtualBoardSection.displayName = 'VirtualBoardSection';
