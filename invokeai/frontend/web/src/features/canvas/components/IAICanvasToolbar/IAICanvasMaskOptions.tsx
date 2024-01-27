import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
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
import IAIColorPicker from 'common/components/IAIColorPicker';
import { canvasMaskSavedToGallery } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  clearMask,
  setIsMaskEnabled,
  setLayer,
  setMaskColor,
  setShouldPreserveMaskedArea,
} from 'features/canvas/store/canvasSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiExcludeBold, PiFloppyDiskBackFill, PiTrashSimpleFill } from 'react-icons/pi';

const formLabelProps: FormLabelProps = {
  flexGrow: 1,
};

const IAICanvasMaskOptions = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const layer = useAppSelector((s) => s.canvas.layer);
  const maskColor = useAppSelector((s) => s.canvas.maskColor);
  const isMaskEnabled = useAppSelector((s) => s.canvas.isMaskEnabled);
  const shouldPreserveMaskedArea = useAppSelector((s) => s.canvas.shouldPreserveMaskedArea);
  const isStaging = useAppSelector(isStagingSelector);

  useHotkeys(
    ['q'],
    () => {
      handleToggleMaskLayer();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [layer]
  );

  useHotkeys(
    ['shift+c'],
    () => {
      handleClearMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  useHotkeys(
    ['h'],
    () => {
      handleToggleEnableMask();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [isMaskEnabled]
  );

  const handleToggleMaskLayer = useCallback(() => {
    dispatch(setLayer(layer === 'mask' ? 'base' : 'mask'));
  }, [dispatch, layer]);

  const handleClearMask = useCallback(() => {
    dispatch(clearMask());
  }, [dispatch]);

  const handleToggleEnableMask = useCallback(() => {
    dispatch(setIsMaskEnabled(!isMaskEnabled));
  }, [dispatch, isMaskEnabled]);

  const handleSaveMask = useCallback(async () => {
    dispatch(canvasMaskSavedToGallery());
  }, [dispatch]);

  const handleChangePreserveMaskedArea = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldPreserveMaskedArea(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeMaskColor = useCallback(
    (newColor: RgbaColor) => {
      dispatch(setMaskColor(newColor));
    },
    [dispatch]
  );

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          aria-label={t('unifiedCanvas.maskingOptions')}
          tooltip={t('unifiedCanvas.maskingOptions')}
          icon={<PiExcludeBold />}
          isChecked={layer === 'mask'}
          isDisabled={isStaging}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControlGroup formLabelProps={formLabelProps}>
              <FormControl>
                <FormLabel>{`${t('unifiedCanvas.enableMask')} (H)`}</FormLabel>
                <Checkbox isChecked={isMaskEnabled} onChange={handleToggleEnableMask} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('unifiedCanvas.preserveMaskedArea')}</FormLabel>
                <Checkbox isChecked={shouldPreserveMaskedArea} onChange={handleChangePreserveMaskedArea} />
              </FormControl>
            </FormControlGroup>
            <Box pt={2} pb={2}>
              <IAIColorPicker color={maskColor} onChange={handleChangeMaskColor} />
            </Box>
            <ButtonGroup isAttached={false}>
              <Button size="sm" leftIcon={<PiFloppyDiskBackFill />} onClick={handleSaveMask}>
                {t('unifiedCanvas.saveMask')}
              </Button>
              <Button size="sm" leftIcon={<PiTrashSimpleFill />} onClick={handleClearMask}>
                {t('unifiedCanvas.clearMask')}
              </Button>
            </ButtonGroup>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(IAICanvasMaskOptions);
