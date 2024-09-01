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
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { snapToNearest } from 'features/controlLayers/konva/util';
import { round } from 'lodash-es';
import { computed } from 'nanostores';
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

function mapSliderValueToRawValue(value: number) {
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

function mapRawValueToSliderValue(value: number) {
  if (value <= 100) {
    return ((value - 10) * 40) / 90;
  } else if (value <= 500) {
    return 40 + ((value - 100) * 30) / 400;
  } else {
    return 70 + ((value - 500) * 30) / 1500;
  }
}

function formatSliderValue(value: number) {
  return String(mapSliderValueToRawValue(value));
}

const marks = [
  mapRawValueToSliderValue(10),
  mapRawValueToSliderValue(50),
  mapRawValueToSliderValue(100),
  mapRawValueToSliderValue(500),
  mapRawValueToSliderValue(2000),
];

const sliderDefaultValue = mapRawValueToSliderValue(100);

const snapCandidates = marks.slice(1, marks.length - 1);

export const CanvasToolbarScale = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const scale = useStore(computed(canvasManager.stateApi.$stageAttrs, (attrs) => attrs.scale));
  const [localScale, setLocalScale] = useState(scale * 100);

  const onChangeSlider = useCallback(
    (scale: number) => {
      let snappedScale = scale;
      // Do not snap if shift key is held
      if (!$shift.get()) {
        snappedScale = snapToNearest(scale, snapCandidates, 2);
      }
      const mappedScale = mapSliderValueToRawValue(snappedScale);
      canvasManager.stage.setScale(mappedScale / 100);
    },
    [canvasManager]
  );

  const onBlur = useCallback(() => {
    if (isNaN(Number(localScale))) {
      canvasManager.stage.setScale(1);
      setLocalScale(100);
      return;
    }
    canvasManager.stage.setScale(localScale / 100);
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
    setLocalScale(scale * 100);
  }, [scale]);

  return (
    <Popover>
      <FormControl w="min-content" gap={2}>
        <FormLabel m={0}>{t('controlLayers.zoom')}</FormLabel>
        <PopoverAnchor>
          <NumberInput
            display="flex"
            alignItems="center"
            min={canvasManager.stage.config.MIN_SCALE * 100}
            max={canvasManager.stage.config.MAX_SCALE * 100}
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
            value={mapRawValueToSliderValue(localScale)}
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

CanvasToolbarScale.displayName = 'CanvasToolbarScale';
