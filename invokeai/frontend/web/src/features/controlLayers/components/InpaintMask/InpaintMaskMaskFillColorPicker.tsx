import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { stopPropagation } from 'common/util/stopPropagation';
import { MaskFillStyle } from 'features/controlLayers/components/common/MaskFillStyle';
import { imFillColorChanged, imFillStyleChanged } from 'features/controlLayers/store/canvasV2Slice';
import type { FillStyle } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import type { RgbColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';

export const InpaintMaskMaskFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fill = useAppSelector((s) => s.canvasV2.inpaintMask.fill);
  const onChangeFillColor = useCallback(
    (color: RgbColor) => {
      dispatch(imFillColorChanged({ color }));
    },
    [dispatch]
  );
  const onChangeFillStyle = useCallback(
    (style: FillStyle) => {
      dispatch(imFillStyleChanged({ style }));
    },
    [dispatch]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex
          role="button"
          aria-label={t('controlLayers.maskPreviewColor')}
          borderRadius="full"
          borderWidth={1}
          bg={rgbColorToString(fill.color)}
          w={8}
          h={8}
          tabIndex={-1}
          onDoubleClick={stopPropagation} // double click expands the layer
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <Flex flexDir="column" gap={4}>
            <RgbColorPicker color={fill.color} onChange={onChangeFillColor} withNumberInput />
            <MaskFillStyle style={fill.style} onChange={onChangeFillStyle} />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

InpaintMaskMaskFillColorPicker.displayName = 'InpaintMaskMaskFillColorPicker';
