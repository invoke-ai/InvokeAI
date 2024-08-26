import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { MaskFillStyle } from 'features/controlLayers/components/common/MaskFillStyle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgFillColorChanged, rgFillStyleChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { FillStyle, RgbColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceMaskFillColorPicker = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fill = useAppSelector((s) => selectEntityOrThrow(s.canvasV2.present, entityIdentifier).fill);
  const onChangeFillColor = useCallback(
    (color: RgbColor) => {
      dispatch(rgFillColorChanged({ entityIdentifier, color }));
    },
    [dispatch, entityIdentifier]
  );
  const onChangeFillStyle = useCallback(
    (style: FillStyle) => {
      dispatch(rgFillStyleChanged({ entityIdentifier, style }));
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

RegionalGuidanceMaskFillColorPicker.displayName = 'RegionalGuidanceMaskFillColorPicker';
