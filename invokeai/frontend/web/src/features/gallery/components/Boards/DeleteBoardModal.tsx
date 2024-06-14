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
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectControlAdaptersSlice } from 'features/controlAdapters/store/controlAdaptersSlice';
import { selectCanvasV2Slice } from 'features/controlLayers/store/controlLayersSlice';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { getImageUsage } from 'features/deleteImageModal/store/selectors';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { some } from 'lodash-es';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllImageNamesForBoardQuery } from 'services/api/endpoints/boards';
import { useDeleteBoardAndImagesMutation, useDeleteBoardMutation } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

type Props = {
  boardToDelete?: BoardDTO;
  setBoardToDelete: (board?: BoardDTO) => void;
};

const DeleteBoardModal = (props: Props) => {
  const { boardToDelete, setBoardToDelete } = props;
  const { t } = useTranslation();
  const { currentData: boardImageNames, isFetching: isFetchingBoardNames } = useListAllImageNamesForBoardQuery(
    boardToDelete?.board_id ?? skipToken
  );

  const selectImageUsageSummary = useMemo(
    () =>
      createMemoizedSelector(
        [selectCanvasSlice, selectNodesSlice, selectControlAdaptersSlice, selectCanvasV2Slice],
        (canvas, nodes, controlAdapters, controlLayers) => {
          const allImageUsage = (boardImageNames ?? []).map((imageName) =>
            getImageUsage(canvas, nodes, controlAdapters, controlLayers.present, imageName)
          );

          const imageUsageSummary: ImageUsage = {
            isCanvasImage: some(allImageUsage, (i) => i.isCanvasImage),
            isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
            isControlImage: some(allImageUsage, (i) => i.isControlImage),
            isControlLayerImage: some(allImageUsage, (i) => i.isControlLayerImage),
          };

          return imageUsageSummary;
        }
      ),
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
    setBoardToDelete(undefined);
  }, [boardToDelete, deleteBoardOnly, setBoardToDelete]);

  const handleDeleteBoardAndImages = useCallback(() => {
    if (!boardToDelete) {
      return;
    }
    deleteBoardAndImages(boardToDelete.board_id);
    setBoardToDelete(undefined);
  }, [boardToDelete, deleteBoardAndImages, setBoardToDelete]);

  const handleClose = useCallback(() => {
    setBoardToDelete(undefined);
  }, [setBoardToDelete]);

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
            {t('controlnet.delete')} {boardToDelete.board_name}
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
