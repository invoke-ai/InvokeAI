import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import {
  setShouldAntialias,
  setShouldAutoSave,
  setShouldCropToBoundingBoxOnSave,
  setShouldDarkenOutsideBoundingBox,
  setShouldRestrictStrokesToBox,
  setShouldShowCanvasDebugInfo,
  setShouldShowGrid,
  setShouldShowIntermediates,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import { isEqual } from 'lodash-es';

import { ChangeEvent, memo, useCallback } from 'react';
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
      shouldAntialias,
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
      shouldAntialias,
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
    shouldAntialias,
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

  const handleChangeShouldSnapToGrid = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldSnapToGrid(e.target.checked)),
    [dispatch]
  );

  const handleChangeShouldShowIntermediates = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldShowIntermediates(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldShowGrid = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldShowGrid(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldDarkenOutsideBoundingBox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldDarkenOutsideBoundingBox(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldAutoSave = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldAutoSave(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldCropToBoundingBoxOnSave = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldCropToBoundingBoxOnSave(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldRestrictStrokesToBox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldRestrictStrokesToBox(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldShowCanvasDebugInfo = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldShowCanvasDebugInfo(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldAntialias = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldAntialias(e.target.checked)),
    [dispatch]
  );

  return (
    <IAIPopover
      isLazy={false}
      triggerComponent={
        <IAIIconButton
          tooltip={t('unifiedCanvas.canvasSettings')}
          aria-label={t('unifiedCanvas.canvasSettings')}
          icon={<FaWrench />}
        />
      }
    >
      <Flex direction="column" gap={2}>
        <IAISimpleCheckbox
          label={t('unifiedCanvas.showIntermediates')}
          isChecked={shouldShowIntermediates}
          onChange={handleChangeShouldShowIntermediates}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.showGrid')}
          isChecked={shouldShowGrid}
          onChange={handleChangeShouldShowGrid}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.snapToGrid')}
          isChecked={shouldSnapToGrid}
          onChange={handleChangeShouldSnapToGrid}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.darkenOutsideSelection')}
          isChecked={shouldDarkenOutsideBoundingBox}
          onChange={handleChangeShouldDarkenOutsideBoundingBox}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.autoSaveToGallery')}
          isChecked={shouldAutoSave}
          onChange={handleChangeShouldAutoSave}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.saveBoxRegionOnly')}
          isChecked={shouldCropToBoundingBoxOnSave}
          onChange={handleChangeShouldCropToBoundingBoxOnSave}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.limitStrokesToBox')}
          isChecked={shouldRestrictStrokesToBox}
          onChange={handleChangeShouldRestrictStrokesToBox}
        />
        <IAISimpleCheckbox
          label={t('unifiedCanvas.showCanvasDebugInfo')}
          isChecked={shouldShowCanvasDebugInfo}
          onChange={handleChangeShouldShowCanvasDebugInfo}
        />

        <IAISimpleCheckbox
          label={t('unifiedCanvas.antialiasing')}
          isChecked={shouldAntialias}
          onChange={handleChangeShouldAntialias}
        />
        <ClearCanvasHistoryButtonModal />
      </Flex>
    </IAIPopover>
  );
};

export default memo(IAICanvasSettingsButtonPopover);
