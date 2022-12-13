import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  setShouldAutoSave,
  setShouldCropToBoundingBoxOnSave,
  setShouldShowCanvasDebugInfo,
  setShouldShowIntermediates,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaWrench } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import EmptyTempFolderButtonModal from 'features/system/components/ClearTempFolderButtonModal';

import ClearCanvasHistoryButtonModal from 'features/canvas/components/ClearCanvasHistoryButtonModal';

export const canvasControlsSelector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      shouldAutoSave,
      shouldCropToBoundingBoxOnSave,
      shouldShowCanvasDebugInfo,
      shouldShowIntermediates,
    } = canvas;

    return {
      shouldAutoSave,
      shouldCropToBoundingBoxOnSave,
      shouldShowCanvasDebugInfo,
      shouldShowIntermediates,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const UnifiedCanvasSettings = () => {
  const dispatch = useAppDispatch();
  const {
    shouldAutoSave,
    shouldCropToBoundingBoxOnSave,
    shouldShowCanvasDebugInfo,
    shouldShowIntermediates,
  } = useAppSelector(canvasControlsSelector);

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          tooltip="Canvas Settings"
          tooltipProps={{
            placement: 'bottom',
          }}
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
          label="Auto Save to Gallery"
          isChecked={shouldAutoSave}
          onChange={(e) => dispatch(setShouldAutoSave(e.target.checked))}
        />
        <IAICheckbox
          label="Save Box Region Only"
          isChecked={shouldCropToBoundingBoxOnSave}
          onChange={(e) =>
            dispatch(setShouldCropToBoundingBoxOnSave(e.target.checked))
          }
        />
        <IAICheckbox
          label="Show Canvas Debug Info"
          isChecked={shouldShowCanvasDebugInfo}
          onChange={(e) =>
            dispatch(setShouldShowCanvasDebugInfo(e.target.checked))
          }
        />
        <ClearCanvasHistoryButtonModal />
        <EmptyTempFolderButtonModal />
      </Flex>
    </IAIPopover>
  );
};

export default UnifiedCanvasSettings;
