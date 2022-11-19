import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  setShouldAutoSave,
  setShouldDarkenOutsideBoundingBox,
  setShouldShowCanvasDebugInfo,
  setShouldShowGrid,
  setShouldShowIntermediates,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaWrench } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

export const canvasControlsSelector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      shouldDarkenOutsideBoundingBox,
      shouldShowIntermediates,
      shouldShowGrid,
      shouldSnapToGrid,
      shouldAutoSave,
      shouldShowCanvasDebugInfo,
    } = canvas;

    return {
      shouldShowGrid,
      shouldSnapToGrid,
      shouldAutoSave,
      shouldDarkenOutsideBoundingBox,
      shouldShowIntermediates,
      shouldShowCanvasDebugInfo,
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
    shouldShowCanvasDebugInfo,
  } = useAppSelector(canvasControlsSelector);

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
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
        <IAICheckbox
          label="Show Canvas Debug Info"
          isChecked={shouldShowCanvasDebugInfo}
          onChange={(e) =>
            dispatch(setShouldShowCanvasDebugInfo(e.target.checked))
          }
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasSettingsButtonPopover;
