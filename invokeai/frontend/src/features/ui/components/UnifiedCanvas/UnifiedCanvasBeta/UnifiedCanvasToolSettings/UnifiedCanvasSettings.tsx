import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  setShouldAutoSave,
  setShouldCropToBoundingBoxOnSave,
  setShouldShowCanvasDebugInfo,
  setShouldShowIntermediates,
} from 'features/canvas/store/canvasSlice';
import EmptyTempFolderButtonModal from 'features/system/components/ClearTempFolderButtonModal';

import { FaWrench } from 'react-icons/fa';

import ClearCanvasHistoryButtonModal from 'features/canvas/components/ClearCanvasHistoryButtonModal';
import { isEqual } from 'lodash';
import { useTranslation } from 'react-i18next';

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
      resultEqualityCheck: isEqual,
    },
  }
);

const UnifiedCanvasSettings = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
          tooltip={t('unifiedCanvas.canvasSettings')}
          tooltipProps={{
            placement: 'bottom',
          }}
          aria-label={t('unifiedCanvas.canvasSettings')}
          icon={<FaWrench />}
        />
      }
    >
      <Flex direction="column" gap="0.5rem">
        <IAICheckbox
          label={t('unifiedCanvas.showIntermediates')}
          isChecked={shouldShowIntermediates}
          onChange={(e) =>
            dispatch(setShouldShowIntermediates(e.target.checked))
          }
        />
        <IAICheckbox
          label={t('unifiedCanvas.autoSaveToGallery')}
          isChecked={shouldAutoSave}
          onChange={(e) => dispatch(setShouldAutoSave(e.target.checked))}
        />
        <IAICheckbox
          label={t('unifiedCanvas.saveBoxRegionOnly')}
          isChecked={shouldCropToBoundingBoxOnSave}
          onChange={(e) =>
            dispatch(setShouldCropToBoundingBoxOnSave(e.target.checked))
          }
        />
        <IAICheckbox
          label={t('unifiedCanvas.showCanvasDebugInfo')}
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
