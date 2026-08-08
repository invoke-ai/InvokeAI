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
import {
  selectSystemShouldConfirmOnDelete,
  selectSystemShouldProtectStarredMedia,
} from 'features/system/store/systemSlice';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetGalleryItemNamesQuery } from 'services/api/endpoints/gallery';
import {
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
  useDeleteUncategorizedImagesMutation,
} from 'services/api/endpoints/images';
import { useDeleteUncategorizedVideosMutation } from 'services/api/endpoints/videos';
import type { BoardDTO } from 'services/api/types';

import { getMediaDeletionSummary } from './getMediaDeletionSummary';

export const $boardToDelete = atom<BoardDTO | 'none' | null>(null);

const DeleteBoardModal = () => {
  useAssertSingleton('DeleteBoardModal');
  const boardToDelete = useStore($boardToDelete);
  const { t } = useTranslation();
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);
  const shouldProtectStarredMedia = useAppSelector(selectSystemShouldProtectStarredMedia);

  const boardId = useMemo(() => (boardToDelete === 'none' ? 'none' : boardToDelete?.board_id), [boardToDelete]);
  const { currentData: boardMedia, isFetching: isFetchingBoardNames } = useGetGalleryItemNamesQuery(
    boardId
      ? {
          board_id: boardId,
          categories: undefined,
          is_intermediate: undefined,
          starred_first: true,
        }
      : skipToken
  );
  const boardImageNames = useMemo(
    () => boardMedia?.items.filter((item) => item.kind === 'image').map((item) => item.name) ?? [],
    [boardMedia?.items]
  );

  const selectImageUsageSummary = useMemo(
    () =>
      createMemoizedSelector(
        [selectNodesSlice, selectCanvasSlice, selectUpscaleSlice, selectRefImagesSlice],
        (nodes, canvas, upscale, refImages) => {
          const allImageUsage = boardImageNames.map((imageName) =>
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
  const [deleteUncategorizedVideos, { isLoading: isDeleteUncategorizedVideosLoading }] =
    useDeleteUncategorizedVideosMutation();

  const imageUsageSummary = useAppSelector(selectImageUsageSummary);
  const [starredConfirmationBoardId, setStarredConfirmationBoardId] = useState<string | null>(null);
  const isStarredConfirmationOpen =
    boardToDelete !== null && boardToDelete !== 'none' && starredConfirmationBoardId === boardToDelete.board_id;

  const handleClose = useCallback(() => {
    setStarredConfirmationBoardId(null);
    $boardToDelete.set(null);
  }, []);

  const reportDeletionSummary = useCallback(
    (summary: ReturnType<typeof getMediaDeletionSummary>, showProtectedWarning: boolean) => {
      if (summary.requestFailed) {
        toast({
          status: 'error',
          title: t('toast.mediaDeleteFailed'),
          description: t('toast.mediaDeleteFailedDesc'),
        });
        return;
      }
      if (summary.failedCount > 0) {
        toast({
          status: 'warning',
          title: t('toast.mediaDeleteFailed'),
          description: t('toast.mediaDeletePartial', { count: summary.failedCount }),
        });
      }
      if (showProtectedWarning && summary.protectedCount > 0) {
        toast({
          status: 'warning',
          title: t('toast.starredMediaProtected'),
          description: t('toast.starredMediaProtectedDesc', { count: summary.protectedCount }),
        });
      }
    },
    [t]
  );

  const handleDeleteBoardOnly = useCallback(async () => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    try {
      await deleteBoardOnly({ board_id: boardToDelete.board_id }).unwrap();
      handleClose();
    } catch {
      toast({
        status: 'error',
        title: t('toast.mediaDeleteFailed'),
        description: t('toast.mediaDeleteFailedDesc'),
      });
    }
  }, [boardToDelete, deleteBoardOnly, handleClose, t]);

  const deleteBoardWithMedia = useCallback(async () => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    const results = await Promise.allSettled([
      deleteBoardAndImages({
        board_id: boardToDelete.board_id,
        delete_starred: !shouldProtectStarredMedia,
      }).unwrap(),
    ]);
    const summary = getMediaDeletionSummary(results);
    reportDeletionSummary(summary, shouldConfirmOnDelete);
    if (!summary.requestFailed) {
      handleClose();
    }
  }, [
    boardToDelete,
    deleteBoardAndImages,
    handleClose,
    reportDeletionSummary,
    shouldConfirmOnDelete,
    shouldProtectStarredMedia,
  ]);

  const handleDeleteBoardAndMedia = useCallback(() => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    if (shouldProtectStarredMedia && (boardMedia?.starred_count ?? 0) > 0) {
      setStarredConfirmationBoardId(boardToDelete.board_id);
      return;
    }
    void deleteBoardWithMedia();
  }, [boardMedia?.starred_count, boardToDelete, deleteBoardWithMedia, shouldProtectStarredMedia]);

  const handleConfirmStarredDelete = useCallback(() => {
    if (!boardToDelete || boardToDelete === 'none' || starredConfirmationBoardId !== boardToDelete.board_id) {
      return;
    }
    void deleteBoardWithMedia();
  }, [boardToDelete, deleteBoardWithMedia, starredConfirmationBoardId]);

  const handleCancelStarredDelete = useCallback(() => {
    setStarredConfirmationBoardId(null);
  }, []);

  const handleDeleteUncategorizedMedia = useCallback(async () => {
    if (!boardToDelete || boardToDelete !== 'none') {
      return;
    }
    const params = shouldProtectStarredMedia ? { delete_starred: false } : undefined;
    const results = await Promise.allSettled([
      deleteUncategorizedImages(params).unwrap(),
      deleteUncategorizedVideos(params).unwrap(),
    ]);
    const summary = getMediaDeletionSummary(results);
    reportDeletionSummary(summary, shouldConfirmOnDelete);
    if (!summary.requestFailed) {
      handleClose();
    }
  }, [
    boardToDelete,
    deleteUncategorizedImages,
    deleteUncategorizedVideos,
    handleClose,
    reportDeletionSummary,
    shouldConfirmOnDelete,
    shouldProtectStarredMedia,
  ]);

  const cancelRef = useRef<HTMLButtonElement>(null);

  const isLoading = useMemo(
    () =>
      isDeleteBoardAndImagesLoading ||
      isDeleteBoardOnlyLoading ||
      isFetchingBoardNames ||
      isDeleteUncategorizedImagesLoading ||
      isDeleteUncategorizedVideosLoading,
    [
      isDeleteBoardAndImagesLoading,
      isDeleteBoardOnlyLoading,
      isFetchingBoardNames,
      isDeleteUncategorizedImagesLoading,
      isDeleteUncategorizedVideosLoading,
    ]
  );

  if (!boardToDelete) {
    return null;
  }

  return (
    <>
      <AlertDialog isOpen={!isStarredConfirmationOpen} onClose={handleClose} leastDestructiveRef={cancelRef} isCentered>
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {t('common.delete')}{' '}
              {boardToDelete === 'none' ? t('boards.uncategorizedImages') : boardToDelete.board_name}
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
                {boardToDelete !== 'none' ? (
                  <Text>{t('boards.deletedBoardsCannotbeRestored')}</Text>
                ) : (
                  <Text>{t('gallery.deleteMediaPermanent')}</Text>
                )}
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
                  <Button colorScheme="error" isLoading={isLoading} onClick={handleDeleteBoardAndMedia}>
                    {t('boards.deleteBoardAndAssets')}
                  </Button>
                )}
                {boardToDelete === 'none' && (
                  <Button colorScheme="error" isLoading={isLoading} onClick={handleDeleteUncategorizedMedia}>
                    {t('boards.deleteAllUncategorizedImages')}
                  </Button>
                )}
              </Flex>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
      <AlertDialog
        isOpen={isStarredConfirmationOpen}
        onClose={handleCancelStarredDelete}
        leastDestructiveRef={cancelRef}
        isCentered
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {t('boards.containsStarredMediaTitle')}
            </AlertDialogHeader>
            <AlertDialogBody>
              <Text>{t('boards.containsStarredMediaConfirm')}</Text>
            </AlertDialogBody>
            <AlertDialogFooter>
              <Flex w="full" gap={2} justifyContent="end">
                <Button ref={cancelRef} onClick={handleCancelStarredDelete}>
                  {t('boards.cancel')}
                </Button>
                <Button colorScheme="error" isLoading={isLoading} onClick={handleConfirmStarredDelete}>
                  {t('common.delete')}
                </Button>
              </Flex>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
};

export default memo(DeleteBoardModal);
