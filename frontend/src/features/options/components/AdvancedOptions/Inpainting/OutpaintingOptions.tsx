import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import { optionsSelector } from 'features/options/store/optionsSelectors';
import {
  setSeamSize,
  setSeamBlur,
  setSeamStrength,
  setSeamSteps,
  setTileSize,
  setShouldForceOutpaint,
} from 'features/options/store/optionsSlice';

const selector = createSelector([optionsSelector], (options) => {
  const {
    seamSize,
    seamBlur,
    seamStrength,
    seamSteps,
    tileSize,
    shouldForceOutpaint,
  } = options;

  return {
    seamSize,
    seamBlur,
    seamStrength,
    seamSteps,
    tileSize,
    shouldForceOutpaint,
  };
});

const OutpaintingOptions = () => {
  const dispatch = useAppDispatch();
  const {
    seamSize,
    seamBlur,
    seamStrength,
    seamSteps,
    tileSize,
    shouldForceOutpaint,
  } = useAppSelector(selector);

  return (
    <Flex direction="column" gap="1rem">
      <IAISlider
        sliderMarkRightOffset={-6}
        label={'Seam Size'}
        min={1}
        max={256}
        sliderNumberInputProps={{ max: 512 }}
        value={seamSize}
        onChange={(v) => {
          dispatch(setSeamSize(v));
        }}
        handleReset={() => dispatch(setSeamSize(96))}
        withInput
        withSliderMarks
        withReset
      />
      <IAISlider
        sliderMarkRightOffset={-4}
        label={'Seam Blur'}
        min={0}
        max={64}
        sliderNumberInputProps={{ max: 512 }}
        value={seamBlur}
        onChange={(v) => {
          dispatch(setSeamBlur(v));
        }}
        handleReset={() => {
          dispatch(setSeamBlur(16));
        }}
        withInput
        withSliderMarks
        withReset
      />
      <IAISlider
        sliderMarkRightOffset={-2}
        label={'Seam Strength'}
        min={0}
        max={1}
        step={0.01}
        value={seamStrength}
        onChange={(v) => {
          dispatch(setSeamStrength(v));
        }}
        handleReset={() => {
          dispatch(setSeamStrength(0.7));
        }}
        withInput
        withSliderMarks
        withReset
      />
      <IAISlider
        sliderMarkRightOffset={-4}
        label={'Seam Steps'}
        min={1}
        max={32}
        sliderNumberInputProps={{ max: 100 }}
        value={seamSteps}
        onChange={(v) => {
          dispatch(setSeamSteps(v));
        }}
        handleReset={() => {
          dispatch(setSeamSteps(10));
        }}
        withInput
        withSliderMarks
        withReset
      />
      <IAISlider
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
      <IAISwitch
        label={'Force Outpaint'}
        isChecked={shouldForceOutpaint}
        onChange={(e) => {
          dispatch(setShouldForceOutpaint(e.target.checked));
        }}
      />
    </Flex>
  );
};

export default OutpaintingOptions;
