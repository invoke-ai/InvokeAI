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
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useRasterLayerOpacity } from 'features/controlLayers/hooks/layerStateHooks';
import { rasterLayerOpacityChanged } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfFill } from 'react-icons/pi';

type Props = {
  layerId: string;
};

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

export const RasterLayerOpacity = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const opacity = useRasterLayerOpacity(layerId);
  const onChangeOpacity = useCallback(
    (v: number) => {
      dispatch(rasterLayerOpacityChanged({ layerId, opacity: v / 100 }));
    },
    [dispatch, layerId]
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

RasterLayerOpacity.displayName = 'RasterLayerOpacity';
