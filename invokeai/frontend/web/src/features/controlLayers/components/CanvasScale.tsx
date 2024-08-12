import {
  $shift,
  CompositeSlider,
  FormControl,
  FormLabel,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from 'features/controlLayers/konva/constants';
import { snapToNearest } from 'features/controlLayers/konva/util';
import { $stageAttrs } from 'features/controlLayers/store/canvasV2Slice';
import { clamp, round } from 'lodash-es';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

function formatPct(v: number | string) {
  if (isNaN(Number(v))) {
    return '';
  }

  return `${round(Number(v), 2).toLocaleString()}%`;
}

function mapSliderValueToScale(value: number) {
  if (value <= 40) {
    // 0 to 40 -> 10% to 100%
    return 10 + (90 * value) / 40;
  } else if (value <= 70) {
    // 40 to 70 -> 100% to 500%
    return 100 + (400 * (value - 40)) / 30;
  } else {
    // 70 to 100 -> 500% to 2000%
    return 500 + (1500 * (value - 70)) / 30;
  }
}

function mapScaleToSliderValue(scale: number) {
  if (scale <= 100) {
    return ((scale - 10) * 40) / 90;
  } else if (scale <= 500) {
    return 40 + ((scale - 100) * 30) / 400;
  } else {
    return 70 + ((scale - 500) * 30) / 1500;
  }
}

function formatSliderValue(value: number) {
  return String(mapSliderValueToScale(value));
}

const marks = [
  mapScaleToSliderValue(10),
  mapScaleToSliderValue(50),
  mapScaleToSliderValue(100),
  mapScaleToSliderValue(500),
  mapScaleToSliderValue(2000),
];

const sliderDefaultValue = mapScaleToSliderValue(100);

const snapCandidates = marks.slice(1, marks.length - 1);

export const CanvasScale = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useStore($canvasManager);
  const stageAttrs = useStore($stageAttrs);
  const [localScale, setLocalScale] = useState(stageAttrs.scale * 100);

  const onChangeSlider = useCallback(
    (scale: number) => {
      if (!canvasManager) {
        return;
      }
      let snappedScale = scale;
      // Do not snap if shift key is held
      if (!$shift.get()) {
        snappedScale = snapToNearest(scale, snapCandidates, 2);
      }
      const mappedScale = mapSliderValueToScale(snappedScale);
      canvasManager.setStageScale(mappedScale / 100);
    },
    [canvasManager]
  );

  const onBlur = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    if (isNaN(Number(localScale))) {
      canvasManager.setStageScale(1);
      setLocalScale(100);
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
    <Popover>
      <FormControl w="min-content" gap={2}>
        <FormLabel m={0}>{t('controlLayers.zoom')}</FormLabel>
        <PopoverAnchor>
          <NumberInput
            display="flex"
            alignItems="center"
            min={MIN_CANVAS_SCALE * 100}
            max={MAX_CANVAS_SCALE * 100}
            value={localScale}
            onChange={onChangeNumberInput}
            onBlur={onBlur}
            w="76px"
            format={formatPct}
            defaultValue={100}
            onKeyDown={onKeyDown}
            clampValueOnBlur={false}
          >
            <NumberInputField paddingInlineEnd={7} />
            <PopoverTrigger>
              <IconButton
                aria-label="open-slider"
                icon={<PiCaretDownBold />}
                size="sm"
                variant="link"
                position="absolute"
                insetInlineEnd={0}
                h="full"
              />
            </PopoverTrigger>
          </NumberInput>
        </PopoverAnchor>
      </FormControl>
      <PopoverContent w={200} pt={0} pb={2} px={4}>
        <PopoverArrow />
        <PopoverBody>
          <CompositeSlider
            min={0}
            max={100}
            value={mapScaleToSliderValue(localScale)}
            onChange={onChangeSlider}
            defaultValue={sliderDefaultValue}
            marks={marks}
            formatValue={formatSliderValue}
            alwaysShowMarks
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

CanvasScale.displayName = 'CanvasScale';
