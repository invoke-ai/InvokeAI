import { ExternalLinkIcon } from '@chakra-ui/icons';
import { MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { selectionAddedToBatch } from 'app/store/middleware/listenerMiddleware/listeners/selectionAddedToBatch';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import {
  imageAddedToBatch,
  imageRemovedFromBatch,
} from 'features/batch/store/batchSlice';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FaExpand,
  FaFolder,
  FaLayerGroup,
  FaShare,
  FaTrash,
} from 'react-icons/fa';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import {
  useAddManyBoardImagesMutation,
  useDeleteBoardImageMutation,
  useDeleteManyBoardImagesMutation,
} from 'services/api/endpoints/boardImages';
import { ImageDTO } from 'services/api/types';
import { AddImageToBoardContext } from '../../../app/contexts/AddImageToBoardContext';
import { sentImageToCanvas, sentImageToImg2Img } from '../store/actions';

type Props = {
  image: ImageDTO;
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const ImageContextMenu = ({ image, children }: Props) => {
  const selector = useMemo(
    () =>
      createSelector(
        [stateSelector],
        ({ gallery, batch }) => {
          const isBatch = gallery.selectedBoardId === 'batch';

          const selection = isBatch ? batch.selection : gallery.selection;
          const isInBatch = batch.ids.includes(image.image_name);

          return { selection, isInBatch };
        },
        defaultSelectorOptions
      ),
    [image.image_name]
  );
  const { selection, isInBatch } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;
  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;

  const { onClickAddToBoard } = useContext(AddImageToBoardContext);

  const handleDelete = useCallback(() => {
    if (!image) {
      return;
    }
    dispatch(imageToDeleteSelected(image));
  }, [dispatch, image]);

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  const [deleteBoardImage] = useDeleteBoardImageMutation();
  const [deleteManyBoardImages] = useDeleteManyBoardImagesMutation();
  const [addManyBoardImages] = useAddManyBoardImagesMutation();

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
    recallBothPrompts(
      image.metadata?.positive_conditioning,
      image.metadata?.negative_conditioning
    );
  }, [
    image.metadata?.negative_conditioning,
    image.metadata?.positive_conditioning,
    recallBothPrompts,
  ]);

  const handleRecallSeed = useCallback(() => {
    recallSeed(image.metadata?.seed);
  }, [image, recallSeed]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(image));
  }, [dispatch, image]);

  // const handleRecallInitialImage = useCallback(() => {
  //   recallInitialImage(image.metadata.invokeai?.node?.image);
  // }, [image, recallInitialImage]);

  const handleSendToCanvas = () => {
    dispatch(sentImageToCanvas());
    dispatch(setInitialCanvasImage(image));
    dispatch(resizeAndScaleCanvas());
    dispatch(setActiveTab('unifiedCanvas'));

    toaster({
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseAllParameters = useCallback(() => {
    recallAllParameters(image);
  }, [image, recallAllParameters]);

  const handleLightBox = () => {
    // dispatch(setCurrentImage(image));
    // dispatch(setIsLightboxOpen(true));
  };

  const handleAddToBoard = useCallback(() => {
    onClickAddToBoard(image);
  }, [image, onClickAddToBoard]);

  const handleRemoveFromBoard = useCallback(() => {
    if (!image.board_id) {
      return;
    }
    deleteBoardImage({ image_name: image.image_name });
  }, [deleteBoardImage, image.board_id, image.image_name]);

  const handleAddSelectionToBoard = useCallback(() => {
    addManyBoardImages({ board_id, image_names: selection });
  }, [addManyBoardImages, selection]);

  const handleRemoveSelectionFromBoard = useCallback(() => {
    deleteManyBoardImages({ image_names: selection });
  }, [deleteManyBoardImages, selection]);

  const handleOpenInNewTab = useCallback(() => {
    window.open(image.image_url, '_blank');
  }, [image.image_url]);

  const handleAddSelectionToBatch = useCallback(() => {
    dispatch(selectionAddedToBatch({ images_names: selection }));
  }, [dispatch, selection]);

  const handleAddToBatch = useCallback(() => {
    dispatch(imageAddedToBatch(image));
  }, [dispatch, image]);

  const handleRemoveFromBatch = useCallback(() => {
    dispatch(imageRemovedFromBatch(image.image_name));
  }, [dispatch, image]);

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      renderMenu={() => (
        <MenuList sx={{ visibility: 'visible !important' }}>
          {selection.length === 1 ? (
            <>
              <MenuItem
                icon={<ExternalLinkIcon />}
                onClickCapture={handleOpenInNewTab}
              >
                {t('common.openInNewTab')}
              </MenuItem>
              {isLightboxEnabled && (
                <MenuItem icon={<FaExpand />} onClickCapture={handleLightBox}>
                  {t('parameters.openInViewer')}
                </MenuItem>
              )}
              <MenuItem
                icon={<IoArrowUndoCircleOutline />}
                onClickCapture={handleRecallPrompt}
                isDisabled={
                  image?.metadata?.positive_conditioning === undefined
                }
              >
                {t('parameters.usePrompt')}
              </MenuItem>

              <MenuItem
                icon={<IoArrowUndoCircleOutline />}
                onClickCapture={handleRecallSeed}
                isDisabled={image?.metadata?.seed === undefined}
              >
                {t('parameters.useSeed')}
              </MenuItem>
              <MenuItem
                icon={<IoArrowUndoCircleOutline />}
                onClickCapture={handleUseAllParameters}
                isDisabled={
                  // what should these be
                  !['t2l', 'l2l', 'inpaint'].includes(
                    String(image?.metadata?.type)
                  )
                }
              >
                {t('parameters.useAll')}
              </MenuItem>
              <MenuItem
                icon={<FaShare />}
                onClickCapture={handleSendToImageToImage}
                id="send-to-img2img"
              >
                {t('parameters.sendToImg2Img')}
              </MenuItem>
              {isCanvasEnabled && (
                <MenuItem
                  icon={<FaShare />}
                  onClickCapture={handleSendToCanvas}
                  id="send-to-canvas"
                >
                  {t('parameters.sendToUnifiedCanvas')}
                </MenuItem>
              )}
              <MenuItem
                icon={<FaLayerGroup />}
                onClickCapture={
                  isInBatch ? handleRemoveFromBatch : handleAddToBatch
                }
              >
                {isInBatch ? 'Remove from Batch' : 'Add to Batch'}
              </MenuItem>
              <MenuItem icon={<FaFolder />} onClickCapture={handleAddToBoard}>
                {image.board_id ? 'Change Board' : 'Add to Board'}
              </MenuItem>
              {image.board_id && (
                <MenuItem
                  icon={<FaFolder />}
                  onClickCapture={handleRemoveFromBoard}
                >
                  Remove from Board
                </MenuItem>
              )}
              <MenuItem
                sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
                icon={<FaTrash />}
                onClickCapture={handleDelete}
              >
                {t('gallery.deleteImage')}
              </MenuItem>
            </>
          ) : (
            <>
              <MenuItem
                icon={<FaFolder />}
                onClickCapture={handleAddSelectionToBoard}
              >
                Move Selection to Board
              </MenuItem>
              <MenuItem
                icon={<FaFolder />}
                onClickCapture={handleRemoveSelectionFromBoard}
              >
                Reset Board for Selection
              </MenuItem>
              <MenuItem
                icon={<FaLayerGroup />}
                onClickCapture={handleAddSelectionToBatch}
              >
                Add Selection to Batch
              </MenuItem>
              <MenuItem
                sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
                icon={<FaTrash />}
                onClickCapture={handleDelete}
              >
                Delete Selection
              </MenuItem>
            </>
          )}
        </MenuList>
      )}
    >
      {children}
    </ContextMenu>
  );
};

export default memo(ImageContextMenu);
