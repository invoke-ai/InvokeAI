import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { stopPropagation } from 'common/util/stopPropagation';
import { MaskFillStyle } from 'features/controlLayers/components/common/MaskFillStyle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgFillColorChanged, rgFillStyleChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectRegionalGuidanceEntityOrThrow } from 'features/controlLayers/store/regionsReducers';
import type { FillStyle, RgbColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceMaskFillColorPicker = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fill = useAppSelector((s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, entityIdentifier.id).fill);
  const onChangeFillColor = useCallback(
    (color: RgbColor) => {
      dispatch(rgFillColorChanged({ id: entityIdentifier.id, color }));
    },
    [dispatch, entityIdentifier.id]
  );
  const onChangeFillStyle = useCallback(
    (style: FillStyle) => {
      dispatch(rgFillStyleChanged({ id: entityIdentifier.id, style }));
    },
    [dispatch, entityIdentifier.id]
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

RegionalGuidanceMaskFillColorPicker.displayName = 'RegionalGuidanceMaskFillColorPicker';
