import { useToast } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIButton from '../../../../common/components/IAIButton';
import {
  InpaintingState,
  setClearBrushHistory,
} from '../../../tabs/Inpainting/inpaintingSlice';
import BoundingBoxSettings from './BoundingBoxSettings';
import _ from 'lodash';

const inpaintingSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { pastLines, futureLines } = inpainting;
    return {
      pastLines,
      futureLines,
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

  const { pastLines, futureLines } = useAppSelector(inpaintingSelector);

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
    </>
  );
}
