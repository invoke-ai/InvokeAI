import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  setBoundingBoxDimensions,
  setScaledBoundingBoxDimensions,
  setShouldScaleBoundingBox,
} from 'features/canvas/store/canvasSlice';
import _ from 'lodash';

const selector = createSelector(
  canvasSelector,
  (canvas) => {
    const {
      boundingBoxDimensions,
      shouldScaleBoundingBox,
      scaledBoundingBoxDimensions,
    } = canvas;
    return {
      boundingBoxDimensions,
      shouldScaleBoundingBox,
      scaledBoundingBoxDimensions,
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
  const {
    boundingBoxDimensions,
    shouldScaleBoundingBox,
    scaledBoundingBoxDimensions,
  } = useAppSelector(selector);

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

  const handleChangeScaledWidth = (v: number) => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        width: Math.floor(v),
      })
    );
  };

  const handleChangeScaledHeight = (v: number) => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        height: Math.floor(v),
      })
    );
  };

  const handleResetScaledWidth = () => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        width: Math.floor(512),
      })
    );
  };

  const handleResetScaledHeight = () => {
    dispatch(
      setScaledBoundingBoxDimensions({
        ...scaledBoundingBoxDimensions,
        height: Math.floor(512),
      })
    );
  };

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
      <IAISwitch
        label={'Scale Image Before Processing'}
        isChecked={shouldScaleBoundingBox}
        onChange={(e) => dispatch(setShouldScaleBoundingBox(e.target.checked))}
      />
      <IAISlider
        isInputDisabled={!shouldScaleBoundingBox}
        isResetDisabled={!shouldScaleBoundingBox}
        isSliderDisabled={!shouldScaleBoundingBox}
        label={'Scaled W'}
        min={64}
        max={1024}
        step={64}
        value={scaledBoundingBoxDimensions.width}
        onChange={handleChangeScaledWidth}
        handleReset={handleResetScaledWidth}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        withReset
      />
      <IAISlider
        isInputDisabled={!shouldScaleBoundingBox}
        isResetDisabled={!shouldScaleBoundingBox}
        isSliderDisabled={!shouldScaleBoundingBox}
        label={'Scaled H'}
        min={64}
        max={1024}
        step={64}
        value={scaledBoundingBoxDimensions.height}
        onChange={handleChangeScaledHeight}
        handleReset={handleResetScaledHeight}
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
