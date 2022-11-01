import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import {
  InpaintingState,
  setInpaintReplace,
  setShouldUseInpaintReplace,
} from '../../../tabs/Inpainting/inpaintingSlice';
import IAISwitch from '../../../../common/components/IAISwitch';

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
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 1rem 0 0.2rem',
      }}
    >
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
  );
}
