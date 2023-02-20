import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
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
import { generationSelector } from 'features/parameters/store/generationSelectors';
import {
  setInfillMethod,
  setTileSize,
} from 'features/parameters/store/generationSlice';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash';

import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [generationSelector, systemSelector, canvasSelector],
  (parameters, system, canvas) => {
    const { tileSize, infillMethod } = parameters;

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
      resultEqualityCheck: isEqual,
    },
  }
);

const InfillAndScalingSettings = () => {
  const dispatch = useAppDispatch();
  const {
    tileSize,
    infillMethod,
    availableInfillMethods,
    boundingBoxScale,
    isManual,
    scaledBoundingBoxDimensions,
  } = useAppSelector(selector);

  const { t } = useTranslation();

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
    <Flex direction="column" gap={4}>
      <IAISelect
        label={t('parameters.scaleBeforeProcessing')}
        validValues={BOUNDING_BOX_SCALES_DICT}
        value={boundingBoxScale}
        onChange={handleChangeBoundingBoxScaleMethod}
      />
      <IAISlider
        isInputDisabled={!isManual}
        isResetDisabled={!isManual}
        isSliderDisabled={!isManual}
        label={t('parameters.scaledWidth')}
        min={64}
        max={1024}
        step={64}
        value={scaledBoundingBoxDimensions.width}
        onChange={handleChangeScaledWidth}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        inputReadOnly
        withReset
        handleReset={handleResetScaledWidth}
        sliderMarkRightOffset={-7}
      />
      <IAISlider
        isInputDisabled={!isManual}
        isResetDisabled={!isManual}
        isSliderDisabled={!isManual}
        label={t('parameters.scaledHeight')}
        min={64}
        max={1024}
        step={64}
        value={scaledBoundingBoxDimensions.height}
        onChange={handleChangeScaledHeight}
        sliderNumberInputProps={{ max: 4096 }}
        withSliderMarks
        withInput
        inputReadOnly
        withReset
        handleReset={handleResetScaledHeight}
        sliderMarkRightOffset={-7}
      />
      <IAISelect
        label={t('parameters.infillMethod')}
        value={infillMethod}
        validValues={availableInfillMethods}
        onChange={(e) => dispatch(setInfillMethod(e.target.value))}
      />
      <IAISlider
        isInputDisabled={infillMethod !== 'tile'}
        isResetDisabled={infillMethod !== 'tile'}
        isSliderDisabled={infillMethod !== 'tile'}
        sliderMarkRightOffset={-4}
        label={t('parameters.tileSize')}
        min={16}
        max={64}
        sliderNumberInputProps={{ max: 256 }}
        value={tileSize}
        onChange={(v) => {
          dispatch(setTileSize(v));
        }}
        withInput
        withSliderMarks
        withReset
        handleReset={() => {
          dispatch(setTileSize(32));
        }}
      />
    </Flex>
  );
};

export default InfillAndScalingSettings;
