import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaEraser } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { InpaintingState, setTool } from '../inpaintingSlice';

import _ from 'lodash';
import { activeTabNameSelector } from '../../../options/optionsSelectors';

const inpaintingEraserSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { tool, shouldShowMask } = inpainting;

    return {
      tool,
      shouldShowMask,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingEraserControl() {
  const { tool, shouldShowMask, activeTabName } = useAppSelector(
    inpaintingEraserSelector
  );
  const dispatch = useAppDispatch();

  const handleSelectEraserTool = () => dispatch(setTool('eraser'));

  // Hotkeys
  // Set tool to eraser
  useHotkeys(
    'e',
    (e: KeyboardEvent) => {
      e.preventDefault();
      if (activeTabName !== 'inpainting' || !shouldShowMask) return;
      handleSelectEraserTool();
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask,
    },
    [activeTabName, shouldShowMask]
  );

  return (
    <IAIIconButton
      aria-label="Eraser (E)"
      tooltip="Eraser (E)"
      icon={<FaEraser />}
      onClick={handleSelectEraserTool}
      data-selected={tool === 'eraser'}
      isDisabled={!shouldShowMask}
    />
  );
}
