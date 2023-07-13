import { ExternalLinkIcon } from '@chakra-ui/icons';
import { MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { imagesAddedToBatch } from 'features/batch/store/batchSlice';
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
import { FaExpand, FaFolder, FaShare, FaTrash } from 'react-icons/fa';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { useRemoveImageFromBoardMutation } from 'services/api/endpoints/boardImages';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
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
        ({ gallery }) => {
          const selectionCount = gallery.selection.length;

          return { selectionCount };
        },
        defaultSelectorOptions
      ),
    []
  );
  const { selectionCount } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const { onClickAddToBoard } = useContext(AddImageToBoardContext);

  const handleDelete = useCallback(() => {
    if (!image) {
      return;
    }
    dispatch(imageToDeleteSelected(image));
  }, [dispatch, image]);

  const handleAddToBoard = useCallback(() => {
    onClickAddToBoard(image);
  }, [image, onClickAddToBoard]);

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      renderMenu={() => (
        <MenuList sx={{ visibility: 'visible !important' }}>
          {selectionCount === 1 ? (
            <SingleSelectionMenuItems image={image} />
          ) : (
            <>
              <MenuItem
                isDisabled={true}
                icon={<FaFolder />}
                onClickCapture={handleAddToBoard}
              >
                Move Selection to Board
              </MenuItem>
              {/* <MenuItem
                icon={<FaFolderPlus />}
                onClickCapture={handleAddSelectionToBatch}
              >
                Add Selection to Batch
              </MenuItem> */}
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

type SingleSelectionMenuItemsProps = {
  image: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { image } = props;

  const selector = useMemo(
    () =>
      createSelector(
        [stateSelector],
        ({ batch }) => {
          const isInBatch = batch.imageNames.includes(image.image_name);

          return { isInBatch };
        },
        defaultSelectorOptions
      ),
    [image.image_name]
  );

  const { isInBatch } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;
  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;

  const { onClickAddToBoard } = useContext(AddImageToBoardContext);

  const { currentData } = useGetImageMetadataQuery(image.image_name);

  const metadata = currentData?.metadata;

  const handleDelete = useCallback(() => {
    if (!image) {
      return;
    }
    dispatch(imageToDeleteSelected(image));
  }, [dispatch, image]);

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  const [removeFromBoard] = useRemoveImageFromBoardMutation();

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
    recallBothPrompts(metadata?.positive_prompt, metadata?.negative_prompt);
  }, [metadata?.negative_prompt, metadata?.positive_prompt, recallBothPrompts]);

  const handleRecallSeed = useCallback(() => {
    recallSeed(metadata?.seed);
  }, [metadata?.seed, recallSeed]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(image));
  }, [dispatch, image]);

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
    console.log(metadata);
    recallAllParameters(metadata);
  }, [metadata, recallAllParameters]);

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
    removeFromBoard({ board_id: image.board_id, image_name: image.image_name });
  }, [image.board_id, image.image_name, removeFromBoard]);

  const handleOpenInNewTab = () => {
    window.open(image.image_url, '_blank');
  };

  const handleAddToBatch = useCallback(() => {
    dispatch(imagesAddedToBatch([image.image_name]));
  }, [dispatch, image.image_name]);

  return (
    <>
      <MenuItem icon={<ExternalLinkIcon />} onClickCapture={handleOpenInNewTab}>
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
          metadata?.positive_prompt === undefined &&
          metadata?.negative_prompt === undefined
        }
      >
        {t('parameters.usePrompt')}
      </MenuItem>

      <MenuItem
        icon={<IoArrowUndoCircleOutline />}
        onClickCapture={handleRecallSeed}
        isDisabled={metadata?.seed === undefined}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={<IoArrowUndoCircleOutline />}
        onClickCapture={handleUseAllParameters}
        isDisabled={!metadata}
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
        icon={<FaFolder />}
        isDisabled={isInBatch}
        onClickCapture={handleAddToBatch}
      >
        Add to Batch
      </MenuItem>
      <MenuItem icon={<FaFolder />} onClickCapture={handleAddToBoard}>
        {image.board_id ? 'Change Board' : 'Add to Board'}
      </MenuItem>
      {image.board_id && (
        <MenuItem icon={<FaFolder />} onClickCapture={handleRemoveFromBoard}>
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
  );
};
