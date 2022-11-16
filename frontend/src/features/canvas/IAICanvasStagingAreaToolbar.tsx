import {
  background,
  ButtonGroup,
  ChakraProvider,
  Flex,
} from '@chakra-ui/react';
import { CacheProvider } from '@emotion/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { emotionCache } from 'main';
import { useCallback, useState } from 'react';
import {
  FaArrowLeft,
  FaArrowRight,
  FaCheck,
  FaEye,
  FaEyeSlash,
  FaTrash,
} from 'react-icons/fa';
import { Group, Rect } from 'react-konva';
import { Html } from 'react-konva-utils';
import {
  commitStagingAreaImage,
  currentCanvasSelector,
  discardStagedImages,
  nextStagingAreaImage,
  prevStagingAreaImage,
} from './canvasSlice';
import IAICanvasImage from './IAICanvasImage';

const selector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    const {
      layerState: {
        stagingArea: { images, selectedImageIndex },
      },
    } = currentCanvas;

    return {
      currentStagingAreaImage:
        images.length > 0 ? images[selectedImageIndex] : undefined,
      isOnFirstImage: selectedImageIndex === 0,
      isOnLastImage: selectedImageIndex === images.length - 1,
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
  const { isOnFirstImage, isOnLastImage, currentStagingAreaImage } =
    useAppSelector(selector);

  const [shouldShowStagedImage, setShouldShowStagedImage] =
    useState<boolean>(true);

  const [shouldShowStagingAreaOutline, setShouldShowStagingAreaOutline] =
    useState<boolean>(true);

  const handleMouseOver = useCallback(() => {
    setShouldShowStagingAreaOutline(false);
  }, []);

  const handleMouseOut = useCallback(() => {
    setShouldShowStagingAreaOutline(true);
  }, []);

  if (!currentStagingAreaImage) return null;

  return (
    <Flex
      pos={'absolute'}
      bottom={'1rem'}
      w={'100%'}
      align={'center'}
      justify={'center'}
    >
      <ButtonGroup isAttached>
        <IAIIconButton
          tooltip="Previous"
          tooltipProps={{ placement: 'bottom' }}
          aria-label="Previous"
          icon={<FaArrowLeft />}
          onClick={() => dispatch(prevStagingAreaImage())}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          data-selected={true}
          isDisabled={isOnFirstImage}
        />
        <IAIIconButton
          tooltip="Next"
          tooltipProps={{ placement: 'bottom' }}
          aria-label="Next"
          icon={<FaArrowRight />}
          onClick={() => dispatch(nextStagingAreaImage())}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          data-selected={true}
          isDisabled={isOnLastImage}
        />
        <IAIIconButton
          tooltip="Accept"
          tooltipProps={{ placement: 'bottom' }}
          aria-label="Accept"
          icon={<FaCheck />}
          onClick={() => dispatch(commitStagingAreaImage())}
          data-selected={true}
        />
        <IAIIconButton
          tooltip="Show/Hide"
          tooltipProps={{ placement: 'bottom' }}
          aria-label="Show/Hide"
          data-alert={!shouldShowStagedImage}
          icon={shouldShowStagedImage ? <FaEye /> : <FaEyeSlash />}
          onClick={() => setShouldShowStagedImage(!shouldShowStagedImage)}
          data-selected={true}
        />
        <IAIIconButton
          tooltip="Discard All"
          tooltipProps={{ placement: 'bottom' }}
          aria-label="Discard All"
          icon={<FaTrash />}
          onClick={() => dispatch(discardStagedImages())}
          data-selected={true}
        />
      </ButtonGroup>
    </Flex>
  );
};

export default IAICanvasStagingAreaToolbar;
