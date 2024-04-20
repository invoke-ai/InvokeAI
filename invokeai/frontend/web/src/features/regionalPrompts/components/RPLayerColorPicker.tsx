import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import {
  isVectorMaskLayer,
  maskLayerPreviewColorChanged,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const RPLayerColorPicker = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const selectColor = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an vector mask layer`);
        return layer.previewColor;
      }),
    [layerId]
  );
  const color = useAppSelector(selectColor);
  const dispatch = useAppDispatch();
  const onColorChange = useCallback(
    (color: RgbaColor) => {
      dispatch(maskLayerPreviewColorChanged({ layerId, color }));
    },
    [dispatch, layerId]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('unifiedCanvas.colorPicker')}
          aria-label={t('unifiedCanvas.colorPicker')}
          size="sm"
          borderRadius="base"
          icon={<PiEyedropperBold />}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <IAIColorPicker color={color} onChange={onColorChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

RPLayerColorPicker.displayName = 'RPLayerColorPicker';
