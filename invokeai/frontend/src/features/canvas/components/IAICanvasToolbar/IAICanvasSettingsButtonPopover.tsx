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
  setShouldDarkenOutsideBoundingBox,
  setShouldRestrictStrokesToBox,
  setShouldShowCanvasDebugInfo,
  setShouldShowGrid,
  setShouldShowIntermediates,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import EmptyTempFolderButtonModal from 'features/system/components/ClearTempFolderButtonModal';
import { isEqual } from 'lodash';

import { ChangeEvent } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaWrench } from 'react-icons/fa';
import ClearCanvasHistoryButtonModal from '../ClearCanvasHistoryButtonModal';

export const canvasControlsSelector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      shouldAutoSave,
      shouldCropToBoundingBoxOnSave,
      shouldDarkenOutsideBoundingBox,
      shouldShowCanvasDebugInfo,
      shouldShowGrid,
      shouldShowIntermediates,
      shouldSnapToGrid,
      shouldRestrictStrokesToBox,
    } = canvas;

    return {
      shouldAutoSave,
      shouldCropToBoundingBoxOnSave,
      shouldDarkenOutsideBoundingBox,
      shouldShowCanvasDebugInfo,
      shouldShowGrid,
      shouldShowIntermediates,
      shouldSnapToGrid,
      shouldRestrictStrokesToBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const IAICanvasSettingsButtonPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    shouldAutoSave,
    shouldCropToBoundingBoxOnSave,
    shouldDarkenOutsideBoundingBox,
    shouldShowCanvasDebugInfo,
    shouldShowGrid,
    shouldShowIntermediates,
    shouldSnapToGrid,
    shouldRestrictStrokesToBox,
  } = useAppSelector(canvasControlsSelector);

  useHotkeys(
    ['n'],
    () => {
      dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
    },
    {
      enabled: true,
      preventDefault: true,
    },
    [shouldSnapToGrid]
  );

  const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldSnapToGrid(e.target.checked));

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          tooltip={t('unifiedCanvas.canvasSettings')}
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
          label={t('unifiedCanvas.showGrid')}
          isChecked={shouldShowGrid}
          onChange={(e) => dispatch(setShouldShowGrid(e.target.checked))}
        />
        <IAICheckbox
          label={t('unifiedCanvas.snapToGrid')}
          isChecked={shouldSnapToGrid}
          onChange={handleChangeShouldSnapToGrid}
        />
        <IAICheckbox
          label={t('unifiedCanvas.darkenOutsideSelection')}
          isChecked={shouldDarkenOutsideBoundingBox}
          onChange={(e) =>
            dispatch(setShouldDarkenOutsideBoundingBox(e.target.checked))
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
          label={t('unifiedCanvas.limitStrokesToBox')}
          isChecked={shouldRestrictStrokesToBox}
          onChange={(e) =>
            dispatch(setShouldRestrictStrokesToBox(e.target.checked))
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

export default IAICanvasSettingsButtonPopover;
