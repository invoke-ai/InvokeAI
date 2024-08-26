import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { fillChanged } from 'features/controlLayers/store/toolSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ToolFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const fill = useAppSelector((s) => s.tool.fill);
  const dispatch = useAppDispatch();
  const onChange = useCallback(
    (color: RgbaColor) => {
      dispatch(fillChanged(color));
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
          bg={rgbaColorToString(fill)}
          w={8}
          h={8}
          cursor="pointer"
          tabIndex={-1}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <IAIColorPicker color={fill} onChange={onChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolFillColorPicker.displayName = 'ToolFillColorPicker';
