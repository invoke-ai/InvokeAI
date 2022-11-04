import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import {
  InpaintingState,
  setInpaintReplace,
  setShouldUseInpaintReplace,
} from '../../../tabs/Inpainting/inpaintingSlice';
import IAISwitch from '../../../../common/components/IAISwitch';
import IAISlider from '../../../../common/components/IAISlider';
import { Flex } from '@chakra-ui/react';

const inpaintReplaceSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { inpaintReplace, shouldUseInpaintReplace } = inpainting;
    return {
      inpaintReplace,
      shouldUseInpaintReplace,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintReplace() {
  const { inpaintReplace, shouldUseInpaintReplace } = useAppSelector(
    inpaintReplaceSelector
  );

  const dispatch = useAppDispatch();

  return (
    <Flex alignItems={'center'} columnGap={'1rem'}>
      <IAISlider
        label="Inpaint Replace"
        value={inpaintReplace}
        onChange={(v: number) => {
          dispatch(setInpaintReplace(v));
        }}
        min={0}
        max={1.0}
        step={0.05}
        isInteger={false}
        isSliderDisabled={!shouldUseInpaintReplace}
        withSliderMarks
        sliderMarkRightOffset={-2}
        withReset
        handleReset={() => dispatch(setInpaintReplace(1))}
        isResetDisabled={!shouldUseInpaintReplace}
      />
      <IAISwitch
        isChecked={shouldUseInpaintReplace}
        onChange={(e: ChangeEvent<HTMLInputElement>) =>
          dispatch(setShouldUseInpaintReplace(e.target.checked))
        }
      />
    </Flex>
  );
}
