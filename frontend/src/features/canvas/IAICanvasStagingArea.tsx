import { background, ButtonGroup, ChakraProvider } from '@chakra-ui/react';
import { CacheProvider } from '@emotion/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { emotionCache } from 'main';
import { useState } from 'react';
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

type Props = GroupConfig;

const IAICanvasStagingArea = (props: Props) => {
  const { ...rest } = props;
  const dispatch = useAppDispatch();
  const { isOnFirstImage, isOnLastImage, currentStagingAreaImage } =
    useAppSelector(selector);

  const [shouldShowStagedImage, setShouldShowStagedImage] =
    useState<boolean>(true);

  if (!currentStagingAreaImage) return null;

  const {
    x,
    y,
    image: { width, height, url },
  } = currentStagingAreaImage;

  return (
    <Group {...rest}>
      <Group>
        {shouldShowStagedImage && <IAICanvasImage url={url} x={x} y={y} />}

        <Rect
          x={x}
          y={y}
          width={width}
          height={height}
          strokeWidth={1}
          stroke={'black'}
          strokeScaleEnabled={false}
        />
        <Rect
          x={x}
          y={y}
          width={width}
          height={height}
          dash={[4, 4]}
          strokeWidth={1}
          stroke={'white'}
          strokeScaleEnabled={false}
        />
      </Group>
      <Html>
        <CacheProvider value={emotionCache}>
          <ChakraProvider>
            <div
              style={{
                position: 'absolute',
                top: y + height,
                left: x + width / 2 - 216 / 2,
                padding: '0.5rem',
                filter: 'drop-shadow(0 0.5rem 1rem rgba(0,0,0))',
              }}
            >
              <ButtonGroup isAttached>
                <IAIIconButton
                  tooltip="Previous"
                  tooltipProps={{ placement: 'bottom' }}
                  aria-label="Previous"
                  icon={<FaArrowLeft />}
                  onClick={() => dispatch(prevStagingAreaImage())}
                  data-selected={true}
                  isDisabled={isOnFirstImage}
                />
                <IAIIconButton
                  tooltip="Next"
                  tooltipProps={{ placement: 'bottom' }}
                  aria-label="Next"
                  icon={<FaArrowRight />}
                  onClick={() => dispatch(nextStagingAreaImage())}
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
                  onClick={() =>
                    setShouldShowStagedImage(!shouldShowStagedImage)
                  }
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
            </div>
          </ChakraProvider>
        </CacheProvider>
      </Html>
    </Group>
  );
};

export default IAICanvasStagingArea;
