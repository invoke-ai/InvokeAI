import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useBoardSearchDisclosure } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import { BoardsSettingsPopover } from 'features/gallery/components/Boards/BoardsSettingsPopover';
import { GalleryHeader } from 'features/gallery/components/GalleryHeader';
import { selectBoardSearchText } from 'features/gallery/store/gallerySelectors';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassBold } from 'react-icons/pi';

export const GalleryTopBar = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const boardSearchDisclosure = useBoardSearchDisclosure();

  const onClickBoardSearch = useCallback(() => {
    if (boardSearchText.length) {
      dispatch(boardSearchTextChanged(''));
    }
    boardSearchDisclosure.toggle();
  }, [boardSearchText.length, boardSearchDisclosure, dispatch]);

  return (
    <Flex alignItems="center" justifyContent="space-between" w="full">
      <Flex flexGrow={1} flexBasis={0}>
        <Text>Boards</Text>
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
          onClick={onClickBoardSearch}
          tooltip={
            boardSearchDisclosure.isOpen ? `${t('gallery.exitBoardSearch')}` : `${t('gallery.displayBoardSearch')}`
          }
          aria-label={t('gallery.displayBoardSearch')}
          icon={<PiMagnifyingGlassBold />}
          colorScheme={boardSearchDisclosure.isOpen ? 'invokeBlue' : 'base'}
        />
      </Flex>
    </Flex>
  );
});
GalleryTopBar.displayName = 'GalleryTopBar';
