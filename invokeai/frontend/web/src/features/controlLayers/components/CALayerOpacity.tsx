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
import { useLayerOpacity } from 'features/controlLayers/hooks/layerStateHooks';
import { layerOpacityChanged } from 'features/controlLayers/store/regionalPromptsSlice';
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
  const opacity = useLayerOpacity(layerId);
  const onChange = useCallback(
    (v: number) => {
      dispatch(layerOpacityChanged({ layerId, opacity: v / 100 }));
    },
    [dispatch, layerId]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          aria-label={t('regionalPrompts.opacity')}
          size="sm"
          icon={<PiDropHalfFill size={16} />}
          variant="ghost"
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl orientation="horizontal" minW={96}>
              <FormLabel m={0}>{t('regionalPrompts.opacity')}</FormLabel>
              <CompositeSlider
                min={0}
                max={100}
                step={1}
                value={opacity}
                defaultValue={100}
                onChange={onChange}
                marks={marks}
              />
              <CompositeNumberInput
                min={0}
                max={100}
                step={1}
                value={opacity}
                defaultValue={100}
                onChange={onChange}
                minW={24}
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
