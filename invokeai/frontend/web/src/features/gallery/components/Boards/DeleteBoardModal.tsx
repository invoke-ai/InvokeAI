import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Flex,
  Skeleton,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import ImageUsageMessage from 'features/deleteImageModal/components/ImageUsageMessage';
import { getImageUsage } from 'features/deleteImageModal/store/selectors';
import { ImageUsage } from 'features/deleteImageModal/store/types';
import { some } from 'lodash-es';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllImageNamesForBoardQuery } from 'services/api/endpoints/boards';
import {
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
} from 'services/api/endpoints/images';
import { BoardDTO } from 'services/api/types';

type Props = {
  boardToDelete?: BoardDTO;
  setBoardToDelete: (board?: BoardDTO) => void;
};

const DeleteBoardModal = (props: Props) => {
  const { boardToDelete, setBoardToDelete } = props;
  const { t } = useTranslation();
  const canRestoreDeletedImagesFromBin = useAppSelector(
    (state) => state.config.canRestoreDeletedImagesFromBin
  );
  const { currentData: boardImageNames, isFetching: isFetchingBoardNames } =
    useListAllImageNamesForBoardQuery(boardToDelete?.board_id ?? skipToken);

  const selectImageUsageSummary = useMemo(
    () =>
      createSelector([stateSelector], (state) => {
        const allImageUsage = (boardImageNames ?? []).map((imageName) =>
          getImageUsage(state, imageName)
        );

        const imageUsageSummary: ImageUsage = {
          isInitialImage: some(allImageUsage, (i) => i.isInitialImage),
          isCanvasImage: some(allImageUsage, (i) => i.isCanvasImage),
          isNodesImage: some(allImageUsage, (i) => i.isNodesImage),
          isControlNetImage: some(allImageUsage, (i) => i.isControlNetImage),
        };
        return { imageUsageSummary };
      }),
    [boardImageNames]
  );

  const [deleteBoardOnly, { isLoading: isDeleteBoardOnlyLoading }] =
    useDeleteBoardMutation();

  const [deleteBoardAndImages, { isLoading: isDeleteBoardAndImagesLoading }] =
    useDeleteBoardAndImagesMutation();

  const { imageUsageSummary } = useAppSelector(selectImageUsageSummary);

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
    () =>
      isDeleteBoardAndImagesLoading ||
      isDeleteBoardOnlyLoading ||
      isFetchingBoardNames,
    [
      isDeleteBoardAndImagesLoading,
      isDeleteBoardOnlyLoading,
      isFetchingBoardNames,
    ]
  );

  if (!boardToDelete) {
    return null;
  }

  return (
    <AlertDialog
      isOpen={Boolean(boardToDelete)}
      onClose={handleClose}
      leastDestructiveRef={cancelRef}
      isCentered
    >
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            Delete {boardToDelete.board_name}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Flex direction="column" gap={3}>
              {isFetchingBoardNames ? (
                <Skeleton>
                  <Flex
                    sx={{
                      w: 'full',
                      h: 32,
                    }}
                  />
                </Skeleton>
              ) : (
                <ImageUsageMessage
                  imageUsage={imageUsageSummary}
                  topMessage="This board contains images used in the following features:"
                  bottomMessage="Deleting this board and its images will reset any features currently using them."
                />
              )}
              <Text>Deleted boards cannot be restored.</Text>
              <Text>
                {canRestoreDeletedImagesFromBin
                  ? t('gallery.deleteImageBin')
                  : t('gallery.deleteImagePermanent')}
              </Text>
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <Flex
              sx={{ justifyContent: 'space-between', width: 'full', gap: 2 }}
            >
              <IAIButton ref={cancelRef} onClick={handleClose}>
                Cancel
              </IAIButton>
              <IAIButton
                colorScheme="warning"
                isLoading={isLoading}
                onClick={handleDeleteBoardOnly}
              >
                Delete Board Only
              </IAIButton>
              <IAIButton
                colorScheme="error"
                isLoading={isLoading}
                onClick={handleDeleteBoardAndImages}
              >
                Delete Board and Images
              </IAIButton>
            </Flex>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteBoardModal);
