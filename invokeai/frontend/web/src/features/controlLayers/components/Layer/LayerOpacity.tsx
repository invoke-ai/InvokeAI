import {
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { layerOpacityChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectLayerOrThrow } from 'features/controlLayers/store/layersReducers';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfFill } from 'react-icons/pi';

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

export const LayerOpacity = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const opacity = useAppSelector((s) => Math.round(selectLayerOrThrow(s.canvasV2, entityIdentifier.id).opacity * 100));
  const onChangeOpacity = useCallback(
    (v: number) => {
      dispatch(layerOpacityChanged({ id: entityIdentifier.id, opacity: v / 100 }));
    },
    [dispatch, entityIdentifier.id]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          aria-label={t('controlLayers.opacity')}
          size="sm"
          icon={<PiDropHalfFill size={16} />}
          variant="ghost"
          onDoubleClick={stopPropagation}
        />
      </PopoverTrigger>
      <PopoverContent onDoubleClick={stopPropagation}>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl orientation="horizontal">
              <FormLabel m={0}>{t('controlLayers.opacity')}</FormLabel>
              <CompositeSlider
                min={0}
                max={100}
                step={1}
                value={opacity}
                defaultValue={100}
                onChange={onChangeOpacity}
                marks={marks}
                w={48}
              />
              <CompositeNumberInput
                min={0}
                max={100}
                step={1}
                value={opacity}
                defaultValue={100}
                onChange={onChangeOpacity}
                w={24}
                format={formatPct}
              />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

LayerOpacity.displayName = 'LayerOpacity';
