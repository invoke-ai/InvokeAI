import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { MaskFillStyle } from 'features/controlLayers/components/common/MaskFillStyle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { inpaintMaskFillColorChanged, inpaintMaskFillStyleChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectInpaintMaskEntityOrThrow } from 'features/controlLayers/store/inpaintMaskReducers';
import type { FillStyle, RgbColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const InpaintMaskMaskFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const fill = useAppSelector((s) => selectInpaintMaskEntityOrThrow(s.canvasV2, entityIdentifier.id).fill);

  const onChangeFillColor = useCallback(
    (color: RgbColor) => {
      dispatch(inpaintMaskFillColorChanged({ entityIdentifier, color }));
    },
    [dispatch, entityIdentifier]
  );
  const onChangeFillStyle = useCallback(
    (style: FillStyle) => {
      dispatch(inpaintMaskFillStyleChanged({ entityIdentifier, style }));
    },
    [dispatch, entityIdentifier]
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
          w="22px"
          h="22px"
          tabIndex={-1}
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
