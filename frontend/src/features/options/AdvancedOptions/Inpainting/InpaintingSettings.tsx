import { useToast } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIButton from '../../../../common/components/IAIButton';
import {
  InpaintingState,
  setClearBrushHistory,
  setInpaintReplace,
  setShouldUseInpaintReplace,
} from '../../../tabs/Inpainting/inpaintingSlice';
import BoundingBoxSettings from './BoundingBoxSettings';
import _ from 'lodash';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import IAISwitch from '../../../../common/components/IAISwitch';

const inpaintingSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { pastLines, futureLines, inpaintReplace, shouldUseInpaintReplace } =
      inpainting;
    return {
      pastLines,
      futureLines,
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

export default function InpaintingSettings() {
  const dispatch = useAppDispatch();
  const toast = useToast();

  const { pastLines, futureLines, inpaintReplace, shouldUseInpaintReplace } =
    useAppSelector(inpaintingSelector);

  const handleClearBrushHistory = () => {
    dispatch(setClearBrushHistory());
    toast({
      title: 'Brush Stroke History Cleared',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };
  return (
    <>
      <BoundingBoxSettings />
      <IAIButton
        label="Clear Brush History"
        onClick={handleClearBrushHistory}
        tooltip="Clears brush stroke history"
        disabled={futureLines.length > 0 || pastLines.length > 0 ? false : true}
      />
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <IAINumberInput
          label="Inpaint Replace"
          value={inpaintReplace}
          min={0}
          max={1.0}
          step={0.05}
          width={'auto'}
          formControlProps={{ style: { paddingRight: '1rem' } }}
          isInteger={false}
          isDisabled={!shouldUseInpaintReplace}
          onChange={(v: number) => {
            dispatch(setInpaintReplace(v));
          }}
        />
        <IAISwitch
          isChecked={shouldUseInpaintReplace}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            dispatch(setShouldUseInpaintReplace(e.target.checked))
          }
        />
      </div>
    </>
  );
}
