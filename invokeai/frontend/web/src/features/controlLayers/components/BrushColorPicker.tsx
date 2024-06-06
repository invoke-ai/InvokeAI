import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { brushColorChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const BrushColorPicker = memo(() => {
  const { t } = useTranslation();
  const brushColor = useAppSelector((s) => s.controlLayers.present.brushColor);
  const dispatch = useAppDispatch();
  const onChange = useCallback(
    (color: RgbaColor) => {
      dispatch(brushColorChanged(color));
    },
    [dispatch]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex
          as="button"
          aria-label={t('controlLayers.brushColor')}
          borderRadius="full"
          borderWidth={1}
          bg={rgbaColorToString(brushColor)}
          w={8}
          h={8}
          cursor="pointer"
          tabIndex={-1}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <IAIColorPicker color={brushColor} onChange={onChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

BrushColorPicker.displayName = 'BrushColorPicker';
