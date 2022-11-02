import { useToast } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
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
import _ from 'lodash';

const clearBrushHistorySelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { pastLines, futureLines } = inpainting;
    return {
      mayClearBrushHistory:
        futureLines.length > 0 || pastLines.length > 0 ? false : true,
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
