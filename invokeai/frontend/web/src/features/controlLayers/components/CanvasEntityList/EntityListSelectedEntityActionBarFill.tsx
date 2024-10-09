import { Box, Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/ColorPicker/RgbColorPicker';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { MaskFillStyle } from 'features/controlLayers/components/common/MaskFillStyle';
import { entityFillColorChanged, entityFillStyleChanged } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityFill, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { type FillStyle, isMaskEntityIdentifier, type RgbColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const EntityListSelectedEntityActionBarFill = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const fill = useAppSelector(selectSelectedEntityFill);

  const onChangeFillColor = useCallback(
    (color: RgbColor) => {
      if (!selectedEntityIdentifier) {
        return;
      }
      if (!isMaskEntityIdentifier(selectedEntityIdentifier)) {
        return;
      }
      dispatch(entityFillColorChanged({ entityIdentifier: selectedEntityIdentifier, color }));
    },
    [dispatch, selectedEntityIdentifier]
  );
  const onChangeFillStyle = useCallback(
    (style: FillStyle) => {
      if (!selectedEntityIdentifier) {
        return;
      }
      if (!isMaskEntityIdentifier(selectedEntityIdentifier)) {
        return;
      }
      dispatch(entityFillStyleChanged({ entityIdentifier: selectedEntityIdentifier, style }));
    },
    [dispatch, selectedEntityIdentifier]
  );

  if (!selectedEntityIdentifier || !fill) {
    return null;
  }

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex role="button" aria-label={t('controlLayers.maskFill')} tabIndex={-1} w={8} h={8}>
          <Tooltip label={t('controlLayers.maskFill')}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbColorToString(fill.color)}
              />
            </Flex>
          </Tooltip>
        </Flex>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <Flex flexDir="column" gap={4}>
            <RgbColorPicker color={fill.color} onChange={onChangeFillColor} withNumberInput withSwatches />
            <MaskFillStyle style={fill.style} onChange={onChangeFillStyle} />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

EntityListSelectedEntityActionBarFill.displayName = 'EntityListSelectedEntityActionBarFill';
