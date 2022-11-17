import { useToast } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIButton from 'common/components/IAIButton';
import {
  canvasSelector,
  setClearBrushHistory,
} from 'features/canvas/canvasSlice';
import _ from 'lodash';

const clearBrushHistorySelector = createSelector(
  canvasSelector,
  (canvas) => {
    const { pastLayerStates, futureLayerStates } = canvas;
    return {
      mayClearBrushHistory:
        futureLayerStates.length > 0 || pastLayerStates.length > 0
          ? false
          : true,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function ClearBrushHistory() {
  const dispatch = useAppDispatch();
  const toast = useToast();

  const { mayClearBrushHistory } = useAppSelector(clearBrushHistorySelector);

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
    <IAIButton
      onClick={handleClearBrushHistory}
      tooltip="Clears brush stroke history"
      disabled={mayClearBrushHistory}
      styleClass="inpainting-options-btn"
    >
      Clear Brush History
    </IAIButton>
  );
}
