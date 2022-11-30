import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISelect from 'common/components/IAISelect';
import IAISlider from 'common/components/IAISlider';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  setBoundingBoxScaleMethod,
  setScaledBoundingBoxDimensions,
} from 'features/canvas/store/canvasSlice';
import {
  BoundingBoxScale,
  BOUNDING_BOX_SCALES_DICT,
} from 'features/canvas/store/canvasTypes';
import { optionsSelector } from 'features/options/store/optionsSelectors';
import {
  setInfillMethod,
  setTileSize,
} from 'features/options/store/optionsSlice';
import { systemSelector } from 'features/system/store/systemSelectors';
import _ from 'lodash';
import { ChangeEvent } from 'react';
import InpaintReplace from './InpaintReplace';

const selector = createSelector(
  [optionsSelector, systemSelector, canvasSelector],
  (options, system, canvas) => {
    const { tileSize, infillMethod } = options;

    const { infill_methods: availableInfillMethods } = system;

    const {
      boundingBoxScaleMethod: boundingBoxScale,
      scaledBoundingBoxDimensions,
    } = canvas;

    return {
      boundingBoxScale,
      scaledBoundingBoxDimensions,
      tileSize,
      infillMethod,
      availableInfillMethods,
      isManual: boundingBoxScale === 'manual',
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const InfillAndScalingOptions = () => {
  const dispatch = useAppDispatch();
  const {
    tileSize,
    infillMethod,
    availableInfillMethods,
    boundingBoxScale,
    isManual,
    scaledBoundingBoxDimensions,
  } = useAppSelector(selector);

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

  const handleChangeBoundingBoxScaleMethod = (
    e: ChangeEvent<HTMLSelectElement>
  ) => {
    dispatch(setBoundingBoxScaleMethod(e.target.value as BoundingBoxScale));
  };

  return (
    <Flex direction="column" gap="1rem">
      <IAISelect
        label={'Scale Before Processing'}
        validValues={BOUNDING_BOX_SCALES_DICT}
        value={boundingBoxScale}
        onChange={handleChangeBoundingBoxScaleMethod}
      />
      <IAISlider
        isInputDisabled={!isManual}
        isResetDisabled={!isManual}
        isSliderDisabled={!isManual}
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
        isInputDisabled={!isManual}
        isResetDisabled={!isManual}
        isSliderDisabled={!isManual}
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
      <InpaintReplace />
      <IAISelect
        label="Infill Method"
        value={infillMethod}
        validValues={availableInfillMethods}
        onChange={(e) => dispatch(setInfillMethod(e.target.value))}
      />
      <IAISlider
        isInputDisabled={infillMethod !== 'tile'}
        isResetDisabled={infillMethod !== 'tile'}
        isSliderDisabled={infillMethod !== 'tile'}
        sliderMarkRightOffset={-4}
        label={'Tile Size'}
        min={16}
        max={64}
        sliderNumberInputProps={{ max: 256 }}
        value={tileSize}
        onChange={(v) => {
          dispatch(setTileSize(v));
        }}
        handleReset={() => {
          dispatch(setTileSize(32));
        }}
        withInput
        withSliderMarks
        withReset
      />
    </Flex>
  );
};

export default InfillAndScalingOptions;
