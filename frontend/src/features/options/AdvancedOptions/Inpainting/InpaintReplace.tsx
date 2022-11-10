import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import IAISwitch from '../../../../common/components/IAISwitch';
import IAISlider from '../../../../common/components/IAISlider';
import { Flex } from '@chakra-ui/react';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setInpaintReplace,
  setShouldUseInpaintReplace,
} from 'features/canvas/canvasSlice';

const canvasInpaintReplaceSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const { inpaintReplace, shouldUseInpaintReplace } = currentCanvas;
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
    canvasInpaintReplaceSelector
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
