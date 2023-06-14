import { Box, Image, MenuItem, MenuList, Text } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback, useState } from 'react';
import { FaImage } from 'react-icons/fa';
import { ContextMenu } from 'chakra-ui-contextmenu';
import { useTranslation } from 'react-i18next';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { useAppToaster } from 'app/components/Toaster';
import { BoardRecord } from 'services/api';
import { EntityId, createSelector } from '@reduxjs/toolkit';
import {
  selectFilteredImagesIds,
  selectImagesById,
} from '../store/imagesSlice';
import { RootState } from '../../../app/store/store';
import { defaultSelectorOptions } from '../../../app/store/util/defaultMemoizeOptions';
import { useSelector } from 'react-redux';

interface HoverableBoardProps {
  board: BoardRecord;
}

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableBoard = memo(({ board }: HoverableBoardProps) => {
  const dispatch = useAppDispatch();

  const { board_name, board_id, cover_image_name } = board;

  const coverImage = useAppSelector((state) =>
    selectImagesById(state, cover_image_name as EntityId)
  );

  const { t } = useTranslation();

  const handleSelectBoard = useCallback(() => {
    // dispatch(imageSelected(board_id));
  }, []);

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <ContextMenu<HTMLDivElement>
        menuProps={{ size: 'sm', isLazy: true }}
        renderMenu={() => (
          <MenuList sx={{ visibility: 'visible !important' }}>
            <MenuItem
              icon={<ExternalLinkIcon />}
              // onClickCapture={handleOpenInNewTab}
            >
              Sample Menu Item
            </MenuItem>
          </MenuList>
        )}
      >
        {(ref) => (
          <Box
            position="relative"
            key={board_id}
            userSelect="none"
            onClick={handleSelectBoard}
            ref={ref}
            sx={{
              display: 'flex',
              flexDir: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              w: 'full',
              h: 'full',
              transition: 'transform 0.2s ease-out',
              aspectRatio: '1/1',
              cursor: 'pointer',
            }}
          >
            <Image
              loading="lazy"
              // objectFit={
              //   shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
              // }
              draggable={false}
              rounded="md"
              src={coverImage ? coverImage.thumbnail_url : undefined}
              fallback={<FaImage />}
              sx={{
                width: '100%',
                height: '100%',
                maxWidth: '100%',
                maxHeight: '100%',
              }}
            />
            <Text textAlign="center">{board_name}</Text>
          </Box>
        )}
      </ContextMenu>
    </Box>
  );
});

HoverableBoard.displayName = 'HoverableBoard';

export default HoverableBoard;
