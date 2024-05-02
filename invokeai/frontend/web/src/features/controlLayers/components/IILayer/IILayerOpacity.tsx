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
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import {
  iiLayerOpacityChanged,
  isInitialImageLayer,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfFill } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

const marks = [0, 25, 50, 75, 100];
const formatPct = (v: number | string) => `${v} %`;

const IILayerOpacity = ({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectOpacity = useMemo(
    () =>
      createSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.filter(isInitialImageLayer).find((l) => l.id === layerId);
        assert(layer, `Layer ${layerId} not found`);
        return Math.round(layer.opacity * 100);
      }),
    [layerId]
  );
  const opacity = useAppSelector(selectOpacity);
  const onChangeOpacity = useCallback(
    (v: number) => {
      dispatch(iiLayerOpacityChanged({ layerId, opacity: v / 100 }));
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

export default memo(IILayerOpacity);
