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
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useLayerOpacity } from 'features/controlLayers/hooks/layerStateHooks';
import { caLayerIsFilterEnabledChanged, layerOpacityChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfFill } from 'react-icons/pi';

type Props = {
  layerId: string;
};

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

const CALayerOpacity = ({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { opacity, isFilterEnabled } = useLayerOpacity(layerId);
  const onChangeOpacity = useCallback(
    (v: number) => {
      dispatch(layerOpacityChanged({ layerId, opacity: v / 100 }));
    },
    [dispatch, layerId]
  );
  const onChangeFilter = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(caLayerIsFilterEnabledChanged({ layerId, isFilterEnabled: e.target.checked }));
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
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl orientation="horizontal" w="full">
              <FormLabel m={0} flexGrow={1} cursor="pointer">
                {t('controlLayers.opacityFilter')}
              </FormLabel>
              <Switch isChecked={isFilterEnabled} onChange={onChangeFilter} />
            </FormControl>
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
};

export default memo(CALayerOpacity);
