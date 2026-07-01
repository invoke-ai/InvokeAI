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
import { some } from 'es-toolkit/compat';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { getImageUsage } from 'features/deleteImageModal/store/state';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { atom } from 'nanostores';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllImageNamesForBoardQuery } from 'services/api/endpoints/boards';
import {
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
  useDeleteUncategorizedImagesMutation,
} from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

export const $boardToDelete = atom<BoardDTO | 'none' | null>(null);

const DeleteBoardModal = () => {
  useAssertSingleton('DeleteBoardModal');
  const boardToDelete = useStore($boardToDelete);
  const { t } = useTranslation();

  const boardId = useMemo(() => (boardToDelete === 'none' ? 'none' : boardToDelete?.board_id), [boardToDelete]);

  const { currentData: boardImageNames, isFetching: isFetchingBoardNames } = useListAllImageNamesForBoardQuery(
    boardId
      ? {
          board_id: boardId,
          categories: undefined,
          is_intermediate: undefined,
        }
      : skipToken
  );

  const selectImageUsageSummary = useMemo(
    () =>
      createMemoizedSelector(
        [selectNodesSlice, selectCanvasSlice, selectUpscaleSlice, selectRefImagesSlice],
        (nodes, canvas, upscale, refImages) => {
          const allImageUsage = (boardImageNames ?? []).map((imageName) =>
            getImageUsage(nodes, canvas, upscale, refImages, imageName)
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
        }
      ),
    [boardImageNames]
  );

  const [deleteBoardOnly, { isLoading: isDeleteBoardOnlyLoading }] = useDeleteBoardMutation();

  const [deleteBoardAndImages, { isLoading: isDeleteBoardAndImagesLoading }] = useDeleteBoardAndImagesMutation();

  const [deleteUncategorizedImages, { isLoading: isDeleteUncategorizedImagesLoading }] =
    useDeleteUncategorizedImagesMutation();

  const imageUsageSummary = useAppSelector(selectImageUsageSummary);

  const handleDeleteBoardOnly = useCallback(() => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    deleteBoardOnly({ board_id: boardToDelete.board_id });
    $boardToDelete.set(null);
  }, [boardToDelete, deleteBoardOnly]);

  const handleDeleteBoardAndImages = useCallback(() => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    deleteBoardAndImages({ board_id: boardToDelete.board_id });
    $boardToDelete.set(null);
  }, [boardToDelete, deleteBoardAndImages]);

  const handleDeleteUncategorizedImages = useCallback(() => {
    if (!boardToDelete || boardToDelete !== 'none') {
      return;
    }
    deleteUncategorizedImages();
    $boardToDelete.set(null);
  }, [boardToDelete, deleteUncategorizedImages]);

  const handleClose = useCallback(() => {
    $boardToDelete.set(null);
  }, []);

  const cancelRef = useRef<HTMLButtonElement>(null);

  const isLoading = useMemo(
    () =>
      isDeleteBoardAndImagesLoading ||
      isDeleteBoardOnlyLoading ||
      isFetchingBoardNames ||
      isDeleteUncategorizedImagesLoading,
    [isDeleteBoardAndImagesLoading, isDeleteBoardOnlyLoading, isFetchingBoardNames, isDeleteUncategorizedImagesLoading]
  );

  if (!boardToDelete) {
    return null;
  }

  return (
    <AlertDialog isOpen={Boolean(boardToDelete)} onClose={handleClose} leastDestructiveRef={cancelRef} isCentered>
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('common.delete')} {boardToDelete === 'none' ? t('boards.uncategorizedImages') : boardToDelete.board_name}
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
              {boardToDelete !== 'none' && <Text>{t('boards.deletedBoardsCannotbeRestored')}</Text>}
              <Text>{t('gallery.deleteImagePermanent')}</Text>
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <Flex w="full" gap={2} justifyContent="end">
              <Button ref={cancelRef} onClick={handleClose}>
                {t('boards.cancel')}
              </Button>
              {boardToDelete !== 'none' && (
                <Button colorScheme="warning" isLoading={isLoading} onClick={handleDeleteBoardOnly}>
                  {t('boards.deleteBoardOnly')}
                </Button>
              )}
              {boardToDelete !== 'none' && (
                <Button colorScheme="error" isLoading={isLoading} onClick={handleDeleteBoardAndImages}>
                  {t('boards.deleteBoardAndImages')}
                </Button>
              )}
              {boardToDelete === 'none' && (
                <Button colorScheme="error" isLoading={isLoading} onClick={handleDeleteUncategorizedImages}>
                  {t('boards.deleteAllUncategorizedImages')}
                </Button>
              )}
            </Flex>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteBoardModal);
