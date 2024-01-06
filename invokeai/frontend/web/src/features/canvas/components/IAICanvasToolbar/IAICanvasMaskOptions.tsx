import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import {
  InvPopover,
  InvPopoverBody,
  InvPopoverContent,
  InvPopoverTrigger,
} from 'common/components/InvPopover/wrapper';
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
import { PiExcludeBold, PiFloppyDiskBackFill, PiTrashSimpleFill } from 'react-icons/pi'

const IAICanvasMaskOptions = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const layer = useAppSelector((s) => s.canvas.layer);
  const maskColor = useAppSelector((s) => s.canvas.maskColor);
  const isMaskEnabled = useAppSelector((s) => s.canvas.isMaskEnabled);
  const shouldPreserveMaskedArea = useAppSelector(
    (s) => s.canvas.shouldPreserveMaskedArea
  );
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
    <InvPopover isLazy>
      <InvPopoverTrigger>
        <InvIconButton
          aria-label={t('unifiedCanvas.maskingOptions')}
          tooltip={t('unifiedCanvas.maskingOptions')}
          icon={<PiExcludeBold />}
          isChecked={layer === 'mask'}
          isDisabled={isStaging}
        />
      </InvPopoverTrigger>
      <InvPopoverContent>
        <InvPopoverBody>
          <Flex direction="column" gap={2}>
            <InvControl label={`${t('unifiedCanvas.enableMask')} (H)`}>
              <InvCheckbox
                isChecked={isMaskEnabled}
                onChange={handleToggleEnableMask}
              />
            </InvControl>
            <InvControl label={t('unifiedCanvas.preserveMaskedArea')}>
              <InvCheckbox
                isChecked={shouldPreserveMaskedArea}
                onChange={handleChangePreserveMaskedArea}
              />
            </InvControl>
            <Box pt={2} pb={2}>
              <IAIColorPicker
                color={maskColor}
                onChange={handleChangeMaskColor}
              />
            </Box>
            <InvButton size="sm" leftIcon={<PiFloppyDiskBackFill />} onClick={handleSaveMask}>
              {t('unifiedCanvas.saveMask')}
            </InvButton>
            <InvButton
              size="sm"
              leftIcon={<PiTrashSimpleFill />}
              onClick={handleClearMask}
            >
              {t('unifiedCanvas.clearMask')}
            </InvButton>
          </Flex>
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
};

export default memo(IAICanvasMaskOptions);
