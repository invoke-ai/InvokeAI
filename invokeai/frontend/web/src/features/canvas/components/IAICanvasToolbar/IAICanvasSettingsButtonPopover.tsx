import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Checkbox,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ClearCanvasHistoryButtonModal from 'features/canvas/components/ClearCanvasHistoryButtonModal';
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
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiGearSixBold } from 'react-icons/pi';

const formLabelProps: FormLabelProps = {
  flexGrow: 1,
};

const IAICanvasSettingsButtonPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shouldAutoSave = useAppSelector((s) => s.canvas.shouldAutoSave);
  const shouldCropToBoundingBoxOnSave = useAppSelector((s) => s.canvas.shouldCropToBoundingBoxOnSave);
  const shouldDarkenOutsideBoundingBox = useAppSelector((s) => s.canvas.shouldDarkenOutsideBoundingBox);
  const shouldShowCanvasDebugInfo = useAppSelector((s) => s.canvas.shouldShowCanvasDebugInfo);
  const shouldShowGrid = useAppSelector((s) => s.canvas.shouldShowGrid);
  const shouldShowIntermediates = useAppSelector((s) => s.canvas.shouldShowIntermediates);
  const shouldSnapToGrid = useAppSelector((s) => s.canvas.shouldSnapToGrid);
  const shouldRestrictStrokesToBox = useAppSelector((s) => s.canvas.shouldRestrictStrokesToBox);
  const shouldAntialias = useAppSelector((s) => s.canvas.shouldAntialias);

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
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldSnapToGrid(e.target.checked)),
    [dispatch]
  );

  const handleChangeShouldShowIntermediates = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldShowIntermediates(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldShowGrid = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldShowGrid(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldDarkenOutsideBoundingBox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldDarkenOutsideBoundingBox(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldAutoSave = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldAutoSave(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldCropToBoundingBoxOnSave = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldCropToBoundingBoxOnSave(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldRestrictStrokesToBox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldRestrictStrokesToBox(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldShowCanvasDebugInfo = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldShowCanvasDebugInfo(e.target.checked)),
    [dispatch]
  );
  const handleChangeShouldAntialias = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldAntialias(e.target.checked)),
    [dispatch]
  );

  return (
    <Popover>
      <PopoverTrigger>
        <IconButton
          tooltip={t('unifiedCanvas.canvasSettings')}
          aria-label={t('unifiedCanvas.canvasSettings')}
          icon={<PiGearSixBold />}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControlGroup formLabelProps={formLabelProps}>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.showIntermediates')}</FormLabel>
                <Checkbox isChecked={shouldShowIntermediates} onChange={handleChangeShouldShowIntermediates} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.showGrid')}</FormLabel>
                <Checkbox isChecked={shouldShowGrid} onChange={handleChangeShouldShowGrid} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.snapToGrid')}</FormLabel>
                <Checkbox isChecked={shouldSnapToGrid} onChange={handleChangeShouldSnapToGrid} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.darkenOutsideSelection')}</FormLabel>
                <Checkbox
                  isChecked={shouldDarkenOutsideBoundingBox}
                  onChange={handleChangeShouldDarkenOutsideBoundingBox}
                />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.autoSaveToGallery')}</FormLabel>
                <Checkbox isChecked={shouldAutoSave} onChange={handleChangeShouldAutoSave} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.saveBoxRegionOnly')}</FormLabel>
                <Checkbox
                  isChecked={shouldCropToBoundingBoxOnSave}
                  onChange={handleChangeShouldCropToBoundingBoxOnSave}
                />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.limitStrokesToBox')}</FormLabel>
                <Checkbox isChecked={shouldRestrictStrokesToBox} onChange={handleChangeShouldRestrictStrokesToBox} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.showCanvasDebugInfo')}</FormLabel>
                <Checkbox isChecked={shouldShowCanvasDebugInfo} onChange={handleChangeShouldShowCanvasDebugInfo} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.antialiasing')}</FormLabel>
                <Checkbox isChecked={shouldAntialias} onChange={handleChangeShouldAntialias} />
              </FormControl>
            </FormControlGroup>
            <ClearCanvasHistoryButtonModal />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(IAICanvasSettingsButtonPopover);
