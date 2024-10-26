import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Flex,
  Skeleton,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { getImageUsage } from 'features/deleteImageModal/store/selectors';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { some } from 'lodash-es';
import { atom } from 'nanostores';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllImageNamesForBoardQuery } from 'services/api/endpoints/boards';
import { useDeleteBoardAndImagesMutation, useDeleteBoardMutation } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

export const $boardToDelete = atom<BoardDTO | null>(null);

const DeleteBoardModal = () => {
  useAssertSingleton('DeleteBoardModal');
  const boardToDelete = useStore($boardToDelete);
  const { t } = useTranslation();
  const { currentData: boardImageNames, isFetching: isFetchingBoardNames } = useListAllImageNamesForBoardQuery(
    boardToDelete?.board_id ?? skipToken
  );

  const selectImageUsageSummary = useMemo(
    () =>
      createMemoizedSelector([selectNodesSlice, selectCanvasSlice, selectUpscaleSlice], (nodes, canvas, upscale) => {
        const allImageUsage = (boardImageNames ?? []).map((imageName) =>
          getImageUsage(nodes, canvas, upscale, imageName)
        );

        const imageUsageSummary: ImageUsage = {
          isUpscaleImage: some(allImageUsage, (i) => i.isUpscaleImage),
          isRasterLayerImage: some(allImageUsage, (i) => i.isRasterLayerImage),
          isInpaintMaskImage: some(allImageUsage, (i) => i.isInpaintMaskImage),
          isRegionalGuidanceImage: some(allImageUsage, (i) => i.isRegionalGuidanceImage),
          isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
          isControlLayerImage: some(allImageUsage, (i) => i.isControlLayerImage),
          isReferenceImage: some(allImageUsage, (i) => i.isReferenceImage),
        };

        return imageUsageSummary;
      }),
    [boardImageNames]
  );

  const [deleteBoardOnly, { isLoading: isDeleteBoardOnlyLoading }] = useDeleteBoardMutation();

  const [deleteBoardAndImages, { isLoading: isDeleteBoardAndImagesLoading }] = useDeleteBoardAndImagesMutation();

  const imageUsageSummary = useAppSelector(selectImageUsageSummary);

  const handleDeleteBoardOnly = useCallback(() => {
    if (!boardToDelete) {
      return;
    }
    deleteBoardOnly(boardToDelete.board_id);
    $boardToDelete.set(null);
  }, [boardToDelete, deleteBoardOnly]);

  const handleDeleteBoardAndImages = useCallback(() => {
    if (!boardToDelete) {
      return;
    }
    deleteBoardAndImages(boardToDelete.board_id);
    $boardToDelete.set(null);
  }, [boardToDelete, deleteBoardAndImages]);

  const handleClose = useCallback(() => {
    $boardToDelete.set(null);
  }, []);

  const cancelRef = useRef<HTMLButtonElement>(null);

  const isLoading = useMemo(
    () => isDeleteBoardAndImagesLoading || isDeleteBoardOnlyLoading || isFetchingBoardNames,
    [isDeleteBoardAndImagesLoading, isDeleteBoardOnlyLoading, isFetchingBoardNames]
  );

  if (!boardToDelete) {
    return null;
  }

  return (
    <AlertDialog isOpen={Boolean(boardToDelete)} onClose={handleClose} leastDestructiveRef={cancelRef} isCentered>
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('common.delete')} {boardToDelete.board_name}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Flex direction="column" gap={3}>
              {isFetchingBoardNames ? (
                <Skeleton>
                  <Flex w="full" h={32} />
                </Skeleton>
              ) : (
                <ImageUsageMessage
                  imageUsage={imageUsageSummary}
                  topMessage={t('boards.topMessage')}
                  bottomMessage={t('boards.bottomMessage')}
                />
              )}
              <Text>
                {boardToDelete.is_private
                  ? t('boards.deletedPrivateBoardsCannotbeRestored')
                  : t('boards.deletedBoardsCannotbeRestored')}
              </Text>
              <Text>{t('gallery.deleteImagePermanent')}</Text>
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <Flex w="full" gap={2} justifyContent="end">
              <Button ref={cancelRef} onClick={handleClose}>
                {t('boards.cancel')}
              </Button>
              <Button colorScheme="warning" isLoading={isLoading} onClick={handleDeleteBoardOnly}>
                {t('boards.deleteBoardOnly')}
              </Button>
              <Button colorScheme="error" isLoading={isLoading} onClick={handleDeleteBoardAndImages}>
                {t('boards.deleteBoardAndImages')}
              </Button>
            </Flex>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteBoardModal);
