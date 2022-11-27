import { ChangeEvent } from 'react';
import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  setBoundingBoxDimensions,
  setGridSnapThreshold,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import _ from 'lodash';
import IAICheckbox from 'common/components/IAICheckbox';
import { useHotkeys } from 'react-hotkeys-hook';

const selector = createSelector(
  canvasSelector,
  (canvas) => {
    const {
      boundingBoxDimensions,
      boundingBoxScaleMethod: boundingBoxScale,
      shouldSnapToGrid,
      gridSnapThreshold,
    } = canvas;
    return {
      shouldSnapToGrid,
      gridSnapThreshold,
      boundingBoxDimensions,
      boundingBoxScale,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const BoundingBoxSettings = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxDimensions, shouldSnapToGrid, gridSnapThreshold } =
    useAppSelector(selector);

  const handleChangeWidth = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(v),
      })
    );
  };

  const handleChangeHeight = (v: number) => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(v),
      })
    );
  };

  const handleResetWidth = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        width: Math.floor(512),
      })
    );
  };

  const handleResetHeight = () => {
    dispatch(
      setBoundingBoxDimensions({
        ...boundingBoxDimensions,
        height: Math.floor(512),
      })
    );
  };

  const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldSnapToGrid(e.target.checked));

  const handleChangeGridSnapThreshold = (v: number) => {
    dispatch(setGridSnapThreshold(v));
  };

  const handleResetGridSnapThreshold = () => {
    dispatch(setGridSnapThreshold(64));
  };

  useHotkeys(
    ['n'],
    () => {
      dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  );

  return (
    <Flex direction="column" gap="1rem">
      <IAISlider
        label={'Width'}
        min={64}
        max={1024}
        step={64}
        value={boundingBoxDimensions.width}
        onChange={handleChangeWidth}
        handleReset={handleResetWidth}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        withReset
      />
      <IAISlider
        label={'Height'}
        min={64}
        max={1024}
        step={64}
        value={boundingBoxDimensions.height}
        onChange={handleChangeHeight}
        handleReset={handleResetHeight}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        withReset
      />
      <IAICheckbox
        label="Snap Bounding Box to Grid (N)"
        isChecked={shouldSnapToGrid}
        onChange={handleChangeShouldSnapToGrid}
      />
      <IAISlider
        label={'Snap Threshold'}
        min={2}
        max={256}
        step={2}
        value={gridSnapThreshold}
        onChange={handleChangeGridSnapThreshold}
        handleReset={handleResetGridSnapThreshold}
        isSliderDisabled={!shouldSnapToGrid}
        isInputDisabled={!shouldSnapToGrid}
        isResetDisabled={!shouldSnapToGrid}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        withReset
      />
    </Flex>
  );
};

export default BoundingBoxSettings;

export const BoundingBoxSettingsHeader = () => {
  return (
    <Box flex="1" textAlign="left">
      Bounding Box
    </Box>
  );
};
