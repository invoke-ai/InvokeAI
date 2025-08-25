import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  ConfirmationAlertDialog,
  Flex,
  Skeleton,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { getStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { some } from 'es-toolkit/compat';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { getImageUsage } from 'features/deleteImageModal/store/state';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { useBoardContainsStarred } from 'features/gallery/hooks/useBoardContainsStarred';
import { selectImageByName } from 'features/gallery/store/gallerySelectors';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import {
  selectSystemShouldConfirmOnDelete,
  selectSystemShouldProtectStarredImages,
} from 'features/system/store/systemSlice';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllImageNamesForBoardQuery } from 'services/api/endpoints/boards';
import {
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
  useDeleteImagesMutation,
  useDeleteUncategorizedImagesMutation,
} from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

export const $boardToDelete = atom<BoardDTO | 'none' | null>(null);

const DeleteBoardModal = () => {
  useAssertSingleton('DeleteBoardModal');
  const boardToDelete = useStore($boardToDelete);
  const { t } = useTranslation();
  // retrieve accidental deletion protection option from app config
  const shouldProtectStarred = useAppSelector(selectSystemShouldProtectStarredImages);

  // we will also need to know if deletion confirmations are enabled
  const shouldConfirmOnDelete = useAppSelector(selectSystemShouldConfirmOnDelete);

  const boardId = useMemo(() => (boardToDelete === 'none' ? 'none' : boardToDelete?.board_id), [boardToDelete]);

  const { isChecking, containsStarred } = useBoardContainsStarred(
    boardToDelete && boardToDelete !== 'none' ? boardId : undefined,
    shouldProtectStarred
  );

  const { isChecking: isCheckingUncategorized, containsStarred: containsStarredUncategorized } =
    useBoardContainsStarred(boardToDelete && boardToDelete === 'none' ? boardId : undefined, shouldProtectStarred);

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

  const [isStarredConfirmOpen, setIsStarredConfirmOpen] = useState(false);
  const pendingDeleteRef = useRef<() => void>(() => {});

  const [deleteBoardOnly, { isLoading: isDeleteBoardOnlyLoading }] = useDeleteBoardMutation();

  const [deleteBoardAndImages, { isLoading: isDeleteBoardAndImagesLoading }] = useDeleteBoardAndImagesMutation();

  const [deleteUncategorizedImages, { isLoading: isDeleteUncategorizedImagesLoading }] =
    useDeleteUncategorizedImagesMutation();

  const [deleteImages, { isLoading: isDeleteImagesLoading }] = useDeleteImagesMutation();

  const imageUsageSummary = useAppSelector(selectImageUsageSummary);

  const handleDeleteBoardOnly = useCallback(() => {
    if (!boardToDelete || boardToDelete === 'none') {
      return;
    }
    deleteBoardOnly({ board_id: boardToDelete.board_id });
    $boardToDelete.set(null);
  }, [boardToDelete, deleteBoardOnly]);

  const performDelete = useCallback(() => {
    if (!boardId) {
      return;
    }
    deleteBoardAndImages({ board_id: boardId });
    $boardToDelete.set(null);
  }, [boardId, deleteBoardAndImages]);

  const finishStarredProtectedDelete = useCallback(() => {
    pendingDeleteRef.current?.();
    setIsStarredConfirmOpen(false);
  }, []);

  const handleDeleteBoardAndImages = useCallback(() => {
    if (!boardId) {
      return;
    }
    if (shouldProtectStarred) {
      if (isChecking) {
        return;
      }
      if (containsStarred) {
        pendingDeleteRef.current = performDelete;
        setIsStarredConfirmOpen(true);
        return;
      }
    }
    performDelete();
  }, [boardId, shouldProtectStarred, isChecking, containsStarred, performDelete]);

  const handleDeleteUncategorizedImages = useCallback(() => {
    if (!boardToDelete || boardToDelete !== 'none') {
      return;
    }

    // here we will check if there are starred images within the "uncategorized" board
    if (shouldProtectStarred) {
      if (isCheckingUncategorized) {
        return; // frontend is still checking, no actions for now
      }

      if (containsStarredUncategorized) {
        const { getState } = getStore();
        const state = getState();

        // now we will sieve through the uncategorized board to separate starred images from the rest
        const starredNames = (boardImageNames ?? []).filter((name) => selectImageByName(state, name)?.starred);

        if (starredNames.length > 0) {
          // the toast should appear only if delete confirmation is enabled, that's the idea
          if (shouldConfirmOnDelete) {
            toast({
              status: 'warning',
              title: t('gallery.cannotDeleteStarred'),
            });
          }

          // now we will delete all the images that are not bearing the star mark
          const namesToDelete = (boardImageNames ?? []).filter((n) => !starredNames.includes(n));

          if (!namesToDelete.length) {
            $boardToDelete.set(null);
            return;
          }

          deleteImages({ image_names: namesToDelete });
          $boardToDelete.set(null);
          return;
        } else {
          // in case all the images are starred, we will only throw a toast. If there's deletion confirmations, that is
          if (shouldConfirmOnDelete) {
            toast({
              status: 'warning',
              title: t('gallery.cannotDeleteStarred'),
            });
          }
          $boardToDelete.set(null);
          return;
        }
      }
    }

    // fallback to standard behavior
    deleteUncategorizedImages();
    $boardToDelete.set(null);
  }, [
    boardToDelete,
    shouldProtectStarred,
    shouldConfirmOnDelete,
    isCheckingUncategorized,
    containsStarredUncategorized,
    boardImageNames,
    deleteImages,
    deleteUncategorizedImages,
    t,
  ]);

  const handleClose = useCallback(() => {
    $boardToDelete.set(null);
  }, []);

  const closeConfirmationAlertDlg = useCallback(() => {
    setIsStarredConfirmOpen(false);
  }, []);

  const cancelRef = useRef<HTMLButtonElement>(null);

  const isLoading = useMemo(
    () =>
      isDeleteBoardAndImagesLoading ||
      isDeleteBoardOnlyLoading ||
      isFetchingBoardNames ||
      isDeleteUncategorizedImagesLoading ||
      isDeleteImagesLoading,
    [
      isDeleteBoardAndImagesLoading,
      isDeleteBoardOnlyLoading,
      isFetchingBoardNames,
      isDeleteUncategorizedImagesLoading,
      isDeleteImagesLoading,
    ]
  );

  if (!boardToDelete) {
    return null;
  }

  let bOpenMainAlertDialog = Boolean(boardToDelete) && !isStarredConfirmOpen;

  return (
    <>
      <AlertDialog isOpen={bOpenMainAlertDialog} onClose={handleClose} leastDestructiveRef={cancelRef} isCentered>
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
                {boardToDelete !== 'none' && (
                  <Text>
                    {boardToDelete.is_private
                      ? t('boards.deletedPrivateBoardsCannotbeRestored')
                      : t('boards.deletedBoardsCannotbeRestored')}
                  </Text>
                )}
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
                  <Button
                    colorScheme="error"
                    isLoading={isLoading || (shouldProtectStarred && isChecking)}
                    onClick={handleDeleteBoardAndImages}
                  >
                    {t('boards.deleteBoardAndImages')}
                  </Button>
                )}
                {boardToDelete === 'none' && (
                  <Button
                    colorScheme="error"
                    isLoading={isLoading || (shouldProtectStarred && isCheckingUncategorized)}
                    onClick={handleDeleteUncategorizedImages}
                  >
                    {t('boards.deleteAllUncategorizedImages')}
                  </Button>
                )}
              </Flex>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
      <ConfirmationAlertDialog
        isOpen={isStarredConfirmOpen}
        onClose={closeConfirmationAlertDlg}
        title={t('boards.containsStarredTitle')}
        acceptCallback={finishStarredProtectedDelete}
        acceptButtonText={t('common.delete')}
        useInert={false}
      >
        <Text>{t('boards.containsStarredConfirm')}</Text>
      </ConfirmationAlertDialog>
    </>
  );
};

export default memo(DeleteBoardModal);
