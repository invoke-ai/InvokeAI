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
import { useCALayerOpacity } from 'features/controlLayers/hooks/layerStateHooks';
import { caFilterChanged, caOpacityChanged } from 'features/controlLayers/store/controlAdaptersSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfFill } from 'react-icons/pi';

type Props = {
  id: string;
};

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

export const CAOpacityAndFilter = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { opacity, isFilterEnabled } = useCALayerOpacity(id);
  const onChangeOpacity = useCallback(
    (v: number) => {
      dispatch(caOpacityChanged({ id, opacity: v / 100 }));
    },
    [dispatch, id]
  );
  const onChangeFilter = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(caFilterChanged({ id, filter: e.target.checked ? 'LightnessToAlphaFilter' : 'none' }));
    },
    [dispatch, id]
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
});

CAOpacityAndFilter.displayName = 'CAOpacityAndFilter';
