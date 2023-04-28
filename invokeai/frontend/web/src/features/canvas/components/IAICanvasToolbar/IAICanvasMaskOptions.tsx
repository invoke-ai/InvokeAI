import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import {
  clearMask,
  setIsMaskEnabled,
  setLayer,
  setMaskColor,
  setShouldPreserveMaskedArea,
} from 'features/canvas/store/canvasSlice';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { isEqual } from 'lodash-es';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaMask, FaTrash } from 'react-icons/fa';

export const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { maskColor, layer, isMaskEnabled, shouldPreserveMaskedArea } =
      canvas;

    return {
      layer,
      maskColor,
      maskColorString: rgbaColorToString(maskColor),
      isMaskEnabled,
      shouldPreserveMaskedArea,
      isStaging,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
const IAICanvasMaskOptions = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    layer,
    maskColor,
    isMaskEnabled,
    shouldPreserveMaskedArea,
    isStaging,
  } = useAppSelector(selector);

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

  const handleToggleMaskLayer = () => {
    dispatch(setLayer(layer === 'mask' ? 'base' : 'mask'));
  };

  const handleClearMask = () => dispatch(clearMask());

  const handleToggleEnableMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));

  return (
    <IAIPopover
      triggerComponent={
        <ButtonGroup>
          <IAIIconButton
            aria-label={t('unifiedCanvas.maskingOptions')}
            tooltip={t('unifiedCanvas.maskingOptions')}
            icon={<FaMask />}
            isChecked={layer === 'mask'}
            isDisabled={isStaging}
          />
        </ButtonGroup>
      }
    >
      <Flex direction="column" gap={2}>
        <IAICheckbox
          label={`${t('unifiedCanvas.enableMask')} (H)`}
          isChecked={isMaskEnabled}
          onChange={handleToggleEnableMask}
        />
        <IAICheckbox
          label={t('unifiedCanvas.preserveMaskedArea')}
          isChecked={shouldPreserveMaskedArea}
          onChange={(e) =>
            dispatch(setShouldPreserveMaskedArea(e.target.checked))
          }
        />
        <IAIColorPicker
          sx={{ paddingTop: 2, paddingBottom: 2 }}
          pickerColor={maskColor}
          onChange={(newColor) => dispatch(setMaskColor(newColor))}
        />
        <IAIButton size="sm" leftIcon={<FaTrash />} onClick={handleClearMask}>
          {t('unifiedCanvas.clearMask')} (Shift+C)
        </IAIButton>
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasMaskOptions;
