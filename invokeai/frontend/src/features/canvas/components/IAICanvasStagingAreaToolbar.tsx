import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import _ from 'lodash';
import { useCallback } from 'react';
import {
  FaArrowLeft,
  FaArrowRight,
  FaCheck,
  FaEye,
  FaEyeSlash,
  FaPlus,
  FaSave,
} from 'react-icons/fa';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  commitStagingAreaImage,
  discardStagedImages,
  nextStagingAreaImage,
  prevStagingAreaImage,
  setShouldShowStagingImage,
  setShouldShowStagingOutline,
} from 'features/canvas/store/canvasSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { saveStagingAreaImageToGallery } from 'app/socketio/actions';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      layerState: {
        stagingArea: { images, selectedImageIndex },
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
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
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

  const handlePrevImage = () => dispatch(prevStagingAreaImage());
  const handleNextImage = () => dispatch(nextStagingAreaImage());
  const handleAccept = () => dispatch(commitStagingAreaImage());

  if (!currentStagingAreaImage) return null;

  return (
    <Flex
      pos={'absolute'}
      bottom={'1rem'}
      w={'100%'}
      align={'center'}
      justify={'center'}
      filter="drop-shadow(0 0.5rem 1rem rgba(0,0,0))"
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <ButtonGroup isAttached>
        <IAIIconButton
          tooltip={`${t('unifiedcanvas:previous')} (Left)`}
          aria-label={`${t('unifiedcanvas:previous')} (Left)`}
          icon={<FaArrowLeft />}
          onClick={handlePrevImage}
          data-selected={true}
          isDisabled={isOnFirstImage}
        />
        <IAIIconButton
          tooltip={`${t('unifiedcanvas:next')} (Right)`}
          aria-label={`${t('unifiedcanvas:next')} (Right)`}
          icon={<FaArrowRight />}
          onClick={handleNextImage}
          data-selected={true}
          isDisabled={isOnLastImage}
        />
        <IAIIconButton
          tooltip={`${t('unifiedcanvas:accept')} (Enter)`}
          aria-label={`${t('unifiedcanvas:accept')} (Enter)`}
          icon={<FaCheck />}
          onClick={handleAccept}
          data-selected={true}
        />
        <IAIIconButton
          tooltip={t('unifiedcanvas:showHide')}
          aria-label={t('unifiedcanvas:showHide')}
          data-alert={!shouldShowStagingImage}
          icon={shouldShowStagingImage ? <FaEye /> : <FaEyeSlash />}
          onClick={() =>
            dispatch(setShouldShowStagingImage(!shouldShowStagingImage))
          }
          data-selected={true}
        />
        <IAIIconButton
          tooltip={t('unifiedcanvas:saveToGallery')}
          aria-label={t('unifiedcanvas:saveToGallery')}
          icon={<FaSave />}
          onClick={() =>
            dispatch(
              saveStagingAreaImageToGallery(currentStagingAreaImage.image.url)
            )
          }
          data-selected={true}
        />
        <IAIIconButton
          tooltip={t('unifiedcanvas:discardAll')}
          aria-label={t('unifiedcanvas:discardAll')}
          icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
          onClick={() => dispatch(discardStagedImages())}
          data-selected={true}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
          fontSize={20}
        />
      </ButtonGroup>
    </Flex>
  );
};

export default IAICanvasStagingAreaToolbar;
