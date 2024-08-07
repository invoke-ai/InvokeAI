import {
  CompositeSlider,
  FormControl,
  FormLabel,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from 'features/controlLayers/konva/constants';
import { $stageAttrs } from 'features/controlLayers/store/canvasV2Slice';
import { clamp, round } from 'lodash-es';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

const formatPct = (v: number | string) => (isNaN(Number(v)) ? '' : `${round(Number(v), 2).toLocaleString()}%`);

export const CanvasScale = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useStore($canvasManager);
  const stageAttrs = useStore($stageAttrs);
  const [localScale, setLocalScale] = useState(stageAttrs.scale * 100);

  const onChange = useCallback(
    (scale: number) => {
      if (!canvasManager) {
        return;
      }
      canvasManager.setStageScale(scale / 100);
    },
    [canvasManager]
  );

  const onReset = useCallback(() => {
    if (!canvasManager) {
      return;
    }

    canvasManager.setStageScale(1);
  }, [canvasManager]);

  const onBlur = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    if (isNaN(Number(localScale))) {
      return;
    }
    canvasManager.setStageScale(clamp(localScale / 100, MIN_CANVAS_SCALE, MAX_CANVAS_SCALE));
  }, [canvasManager, localScale]);

  const onChangeNumberInput = useCallback((valueAsString: string, valueAsNumber: number) => {
    setLocalScale(valueAsNumber);
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      }
    },
    [onBlur]
  );

  useEffect(() => {
    setLocalScale(stageAttrs.scale * 100);
  }, [stageAttrs.scale]);

  return (
    <FormControl w="min-content" gap={2}>
      <FormLabel m={0}>{t('controlLayers.zoom')}</FormLabel>
      <Popover isLazy trigger="hover" openDelay={300}>
        <PopoverTrigger>
          <NumberInput
            min={MIN_CANVAS_SCALE * 100}
            max={MAX_CANVAS_SCALE * 100}
            value={localScale}
            onChange={onChangeNumberInput}
            onBlur={onBlur}
            w="64px"
            format={formatPct}
            defaultValue={100}
            onKeyDown={onKeyDown}
          >
            <NumberInputField textAlign="center" paddingInlineEnd={3} />
          </NumberInput>
        </PopoverTrigger>
        <PopoverContent w={200} py={2} px={4}>
          <PopoverArrow />
          <PopoverBody>
            <CompositeSlider
              min={MIN_CANVAS_SCALE * 100}
              max={MAX_CANVAS_SCALE * 100}
              value={stageAttrs.scale * 100}
              onChange={onChange}
              defaultValue={100}
            />
          </PopoverBody>
        </PopoverContent>
      </Popover>
      <IconButton aria-label="reset" onClick={onReset} icon={<PiArrowCounterClockwiseBold />} variant="link" />
    </FormControl>
  );
});

CanvasScale.displayName = 'CanvasScale';
