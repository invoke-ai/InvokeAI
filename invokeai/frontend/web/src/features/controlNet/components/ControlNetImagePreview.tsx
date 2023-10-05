import { Box, Flex, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { setHeight, setWidth } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaRulerVertical, FaSave, FaUndo } from 'react-icons/fa';
import {
  useAddImageToBoardMutation,
  useChangeImageIsIntermediateMutation,
  useGetImageDTOQuery,
  useRemoveImageFromBoardMutation,
} from 'services/api/endpoints/images';
import { PostUploadAction } from 'services/api/types';
import IAIDndImageIcon from '../../../common/components/IAIDndImageIcon';
import { controlAdapterImageChanged } from '../store/controlAdaptersSlice';
import { useControlAdapterControlImage } from '../hooks/useControlAdapterControlImage';
import { useControlAdapterProcessedControlImage } from '../hooks/useControlAdapterProcessedControlImage';
import { useControlAdapterProcessorType } from '../hooks/useControlAdapterProcessorType';

type Props = {
  id: string;
  isSmall?: boolean;
};

const selector = createSelector(
  stateSelector,
  ({ controlAdapters, gallery }) => {
    const { pendingControlImages } = controlAdapters;
    const { autoAddBoardId } = gallery;

    return {
      pendingControlImages,
      autoAddBoardId,
    };
  },
  defaultSelectorOptions
);

const ControlNetImagePreview = ({ isSmall, id }: Props) => {
  const controlImageName = useControlAdapterControlImage(id);
  const processedControlImageName = useControlAdapterProcessedControlImage(id);
  const processorType = useControlAdapterProcessorType(id);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { pendingControlImages, autoAddBoardId } = useAppSelector(selector);
  const activeTabName = useAppSelector(activeTabNameSelector);

  const [isMouseOverImage, setIsMouseOverImage] = useState(false);

  const { currentData: controlImage } = useGetImageDTOQuery(
    controlImageName ?? skipToken
  );

  const { currentData: processedControlImage } = useGetImageDTOQuery(
    processedControlImageName ?? skipToken
  );

  const [changeIsIntermediate] = useChangeImageIsIntermediateMutation();
  const [addToBoard] = useAddImageToBoardMutation();
  const [removeFromBoard] = useRemoveImageFromBoardMutation();
  const handleResetControlImage = useCallback(() => {
    dispatch(controlAdapterImageChanged({ id, controlImage: null }));
  }, [id, dispatch]);

  const handleSaveControlImage = useCallback(async () => {
    if (!processedControlImage) {
      return;
    }

    await changeIsIntermediate({
      imageDTO: processedControlImage,
      is_intermediate: false,
    }).unwrap();

    if (autoAddBoardId !== 'none') {
      addToBoard({
        imageDTO: processedControlImage,
        board_id: autoAddBoardId,
      });
    } else {
      removeFromBoard({ imageDTO: processedControlImage });
    }
  }, [
    processedControlImage,
    changeIsIntermediate,
    autoAddBoardId,
    addToBoard,
    removeFromBoard,
  ]);

  const handleSetControlImageToDimensions = useCallback(() => {
    if (!controlImage) {
      return;
    }

    if (activeTabName === 'unifiedCanvas') {
      dispatch(
        setBoundingBoxDimensions({
          width: controlImage.width,
          height: controlImage.height,
        })
      );
    } else {
      dispatch(setWidth(controlImage.width));
      dispatch(setHeight(controlImage.height));
    }
  }, [controlImage, activeTabName, dispatch]);

  const handleMouseEnter = useCallback(() => {
    setIsMouseOverImage(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseOverImage(false);
  }, []);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, id]);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      id,
      actionType: 'SET_CONTROL_ADAPTER_IMAGE',
      context: { id },
    }),
    [id]
  );

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({ type: 'SET_CONTROL_ADAPTER_IMAGE', id }),
    [id]
  );

  const shouldShowProcessedImage =
    controlImage &&
    processedControlImage &&
    !isMouseOverImage &&
    !pendingControlImages.includes(id) &&
    processorType !== 'none';

  return (
    <Flex
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      sx={{
        position: 'relative',
        w: 'full',
        h: isSmall ? 28 : 366, // magic no touch
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        isDropDisabled={shouldShowProcessedImage}
        postUploadAction={postUploadAction}
      />

      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          w: 'full',
          h: 'full',
          opacity: shouldShowProcessedImage ? 1 : 0,
          transitionProperty: 'common',
          transitionDuration: 'normal',
          pointerEvents: 'none',
        }}
      >
        <IAIDndImage
          draggableData={draggableData}
          droppableData={droppableData}
          imageDTO={processedControlImage}
          isUploadDisabled={true}
        />
      </Box>

      <>
        <IAIDndImageIcon
          onClick={handleResetControlImage}
          icon={controlImage ? <FaUndo /> : undefined}
          tooltip={t('controlnet.resetControlImage')}
        />
        <IAIDndImageIcon
          onClick={handleSaveControlImage}
          icon={controlImage ? <FaSave size={16} /> : undefined}
          tooltip={t('controlnet.saveControlImage')}
          styleOverrides={{ marginTop: 6 }}
        />
        <IAIDndImageIcon
          onClick={handleSetControlImageToDimensions}
          icon={controlImage ? <FaRulerVertical size={16} /> : undefined}
          tooltip={t('controlnet.setControlImageDimensions')}
          styleOverrides={{ marginTop: 12 }}
        />
      </>

      {pendingControlImages.includes(id) && (
        <Flex
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineStart: 0,
            w: 'full',
            h: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: 0.8,
            borderRadius: 'base',
            bg: 'base.400',
            _dark: {
              bg: 'base.900',
            },
          }}
        >
          <Spinner
            size="xl"
            sx={{ color: 'base.100', _dark: { color: 'base.400' } }}
          />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(ControlNetImagePreview);
