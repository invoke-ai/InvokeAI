import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger, Tooltip } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import {
  isMaskedGuidanceLayer,
  maskLayerPreviewColorChanged,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import type { RgbColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';
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
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an vector mask layer`);
        return layer.previewColor;
      }),
    [layerId]
  );
  const color = useAppSelector(selectColor);
  const dispatch = useAppDispatch();
  const onColorChange = useCallback(
    (color: RgbColor) => {
      dispatch(maskLayerPreviewColorChanged({ layerId, color }));
    },
    [dispatch, layerId]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <span>
          <Tooltip label={t('regionalPrompts.maskPreviewColor')}>
            <Flex
              as="button"
              aria-label={t('regionalPrompts.maskPreviewColor')}
              borderRadius="base"
              borderWidth={1}
              bg={rgbColorToString(color)}
              w={8}
              h={8}
              cursor="pointer"
              tabIndex={-1}
            />
          </Tooltip>
        </span>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <RgbColorPicker color={color} onChange={onColorChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

RPLayerColorPicker.displayName = 'RPLayerColorPicker';
