import {
  Box,
  Flex,
  Icon,
  Image,
  MenuItem,
  MenuList,
  Text,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { PropsWithChildren, memo, useCallback, useState } from 'react';
import { FaFolder, FaImage } from 'react-icons/fa';
import { ContextMenu } from 'chakra-ui-contextmenu';
import { useTranslation } from 'react-i18next';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { useAppToaster } from 'app/components/Toaster';
import { BoardDTO } from 'services/api';
import { EntityId, createSelector } from '@reduxjs/toolkit';
import {
  selectFilteredImagesIds,
  selectImagesById,
} from '../../store/imagesSlice';
import { RootState } from '../../../../app/store/store';
import { defaultSelectorOptions } from '../../../../app/store/util/defaultMemoizeOptions';
import { useSelector } from 'react-redux';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/boardSlice';

interface HoverableBoardProps {
  board: BoardDTO;
}

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableBoard = memo(({ board }: HoverableBoardProps) => {
  const dispatch = useAppDispatch();

  const { board_name, board_id, cover_image_url } = board;

  const { t } = useTranslation();

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(board_id));
  }, [board_id, dispatch]);

  return (
    <Box sx={{ touchAction: 'none' }}>
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
          <Flex
            position="relative"
            key={board_id}
            userSelect="none"
            onClick={handleSelectBoard}
            ref={ref}
            sx={{
              flexDir: 'column',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              w: 'full',
              h: 'full',
              gap: 1,
            }}
          >
            <Flex
              sx={{
                justifyContent: 'center',
                alignItems: 'center',
                borderWidth: '1px',
                borderRadius: 'base',
                borderColor: 'base.800',
                w: 'full',
                h: 'full',
                aspectRatio: '1/1',
              }}
            >
              {cover_image_url ? (
                <Image
                  loading="lazy"
                  objectFit="cover"
                  draggable={false}
                  rounded="md"
                  src={cover_image_url}
                  fallback={<IAIImageFallback />}
                  sx={{}}
                />
              ) : (
                <Icon boxSize={8} color="base.700" as={FaFolder} />
              )}
            </Flex>
            <Text sx={{ color: 'base.200', fontSize: 'xs' }}>{board_name}</Text>
          </Flex>
        )}
      </ContextMenu>
    </Box>
  );
});

HoverableBoard.displayName = 'HoverableBoard';

export default HoverableBoard;
