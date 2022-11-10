import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  currentCanvasSelector,
  outpaintingCanvasSelector,
  setShouldAutoSave,
  setShouldDarkenOutsideBoundingBox,
  setShouldShowGrid,
  setShouldShowIntermediates,
  setShouldSnapToGrid,
} from './canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaWrench } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';

export const canvasControlsSelector = createSelector(
  [currentCanvasSelector, outpaintingCanvasSelector],
  (currentCanvas, outpaintingCanvas) => {
    const { shouldDarkenOutsideBoundingBox, shouldShowIntermediates } =
      currentCanvas;

    const { shouldShowGrid, shouldSnapToGrid, shouldAutoSave } =
      outpaintingCanvas;

    return {
      shouldShowGrid,
      shouldSnapToGrid,
      shouldAutoSave,
      shouldDarkenOutsideBoundingBox,
      shouldShowIntermediates,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasSettingsButtonPopover = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowIntermediates,
    shouldShowGrid,
    shouldSnapToGrid,
    shouldAutoSave,
    shouldDarkenOutsideBoundingBox,
  } = useAppSelector(canvasControlsSelector);

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          variant="link"
          data-variant="link"
          tooltip="Canvas Settings"
          aria-label="Canvas Settings"
          icon={<FaWrench />}
        />
      }
    >
      <Flex direction={'column'} gap={'0.5rem'}>
        <IAICheckbox
          label="Show Intermediates"
          isChecked={shouldShowIntermediates}
          onChange={(e) =>
            dispatch(setShouldShowIntermediates(e.target.checked))
          }
        />
        <IAICheckbox
          label="Show Grid"
          isChecked={shouldShowGrid}
          onChange={(e) => dispatch(setShouldShowGrid(e.target.checked))}
        />
        <IAICheckbox
          label="Snap to Grid"
          isChecked={shouldSnapToGrid}
          onChange={(e) => dispatch(setShouldSnapToGrid(e.target.checked))}
        />
        <IAICheckbox
          label="Darken Outside Selection"
          isChecked={shouldDarkenOutsideBoundingBox}
          onChange={(e) =>
            dispatch(setShouldDarkenOutsideBoundingBox(e.target.checked))
          }
        />
        <IAICheckbox
          label="Auto Save to Gallery"
          isChecked={shouldAutoSave}
          onChange={(e) => dispatch(setShouldAutoSave(e.target.checked))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasSettingsButtonPopover;
