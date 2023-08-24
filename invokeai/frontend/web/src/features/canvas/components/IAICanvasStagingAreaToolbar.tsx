import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  commitStagingAreaImage,
  discardStagedImages,
  nextStagingAreaImage,
  prevStagingAreaImage,
  setShouldShowStagingImage,
  setShouldShowStagingOutline,
} from 'features/canvas/store/canvasSlice';
import { isEqual } from 'lodash-es';

import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaArrowLeft,
  FaArrowRight,
  FaCheck,
  FaEye,
  FaEyeSlash,
  FaPlus,
  FaSave,
} from 'react-icons/fa';
import { stagingAreaImageSaved } from '../store/actions';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { skipToken } from '@reduxjs/toolkit/dist/query';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      layerState: {
        stagingArea: { images, selectedImageIndex, sessionId },
      },
      shouldShowStagingOutline,
      shouldShowStagingImage,
    } = canvas;

    return {
      currentStagingAreaImage:
        images.length > 0 ? images[selectedImageIndex] : undefined,
      isOnFirstImage: selectedImageIndex === 0,
      isOnLastImage: selectedImageIndex === images.length - 1,
      shouldShowStagingImage,
      shouldShowStagingOutline,
      sessionId,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const IAICanvasStagingAreaToolbar = () => {
  const dispatch = useAppDispatch();
  const {
    isOnFirstImage,
    isOnLastImage,
    currentStagingAreaImage,
    shouldShowStagingImage,
    sessionId,
  } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleMouseOver = useCallback(() => {
    dispatch(setShouldShowStagingOutline(true));
  }, [dispatch]);

  const handleMouseOut = useCallback(() => {
    dispatch(setShouldShowStagingOutline(false));
  }, [dispatch]);

  useHotkeys(
    ['left'],
    () => {
      handlePrevImage();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['right'],
    () => {
      handleNextImage();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['enter'],
    () => {
      handleAccept();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  const handlePrevImage = useCallback(
    () => dispatch(prevStagingAreaImage()),
    [dispatch]
  );

  const handleNextImage = useCallback(
    () => dispatch(nextStagingAreaImage()),
    [dispatch]
  );

  const handleAccept = useCallback(
    () => dispatch(commitStagingAreaImage(sessionId)),
    [dispatch, sessionId]
  );

  const { data: imageDTO } = useGetImageDTOQuery(
    currentStagingAreaImage?.imageName ?? skipToken
  );

  if (!currentStagingAreaImage) {
    return null;
  }

  return (
    <Flex
      pos="absolute"
      bottom={4}
      w="100%"
      align="center"
      justify="center"
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <ButtonGroup isAttached borderRadius="base" shadow="dark-lg">
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.previous')} (Left)`}
          aria-label={`${t('unifiedCanvas.previous')} (Left)`}
          icon={<FaArrowLeft />}
          onClick={handlePrevImage}
          colorScheme="accent"
          isDisabled={isOnFirstImage}
        />
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.next')} (Right)`}
          aria-label={`${t('unifiedCanvas.next')} (Right)`}
          icon={<FaArrowRight />}
          onClick={handleNextImage}
          colorScheme="accent"
          isDisabled={isOnLastImage}
        />
        <IAIIconButton
          tooltip={`${t('unifiedCanvas.accept')} (Enter)`}
          aria-label={`${t('unifiedCanvas.accept')} (Enter)`}
          icon={<FaCheck />}
          onClick={handleAccept}
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={t('unifiedCanvas.showHide')}
          aria-label={t('unifiedCanvas.showHide')}
          data-alert={!shouldShowStagingImage}
          icon={shouldShowStagingImage ? <FaEye /> : <FaEyeSlash />}
          onClick={() =>
            dispatch(setShouldShowStagingImage(!shouldShowStagingImage))
          }
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={t('unifiedCanvas.saveToGallery')}
          aria-label={t('unifiedCanvas.saveToGallery')}
          isDisabled={!imageDTO || !imageDTO.is_intermediate}
          icon={<FaSave />}
          onClick={() => {
            if (!imageDTO) {
              return;
            }

            dispatch(
              stagingAreaImageSaved({
                imageDTO,
              })
            );
          }}
          colorScheme="accent"
        />
        <IAIIconButton
          tooltip={t('unifiedCanvas.discardAll')}
          aria-label={t('unifiedCanvas.discardAll')}
          icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
          onClick={() => dispatch(discardStagedImages())}
          colorScheme="error"
          fontSize={20}
        />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(IAICanvasStagingAreaToolbar);
