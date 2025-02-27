import {
  $shift,
  CompositeSlider,
  Flex,
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
import { PiCaretDownBold, PiMagnifyingGlassMinusBold, PiMagnifyingGlassPlusBold } from 'react-icons/pi';

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
  const canvasManager = useCanvasManager();
  const scale = useStore(computed(canvasManager.stage.$stageAttrs, (attrs) => attrs.scale));
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
    <Flex alignItems="center">
      <ZoomOutButton />
      <Popover>
        <PopoverAnchor>
          <NumberInput
            variant="outline"
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
            <NumberInputField paddingInlineEnd={7} title="" _focusVisible={{ zIndex: 0 }} />
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
      <ZoomInButton />
    </Flex>
  );
});

CanvasToolbarScale.displayName = 'CanvasToolbarScale';

const SCALE_SNAPS = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 7.5, 10, 15, 20];

const ZoomOutButton = () => {
  const canvasManager = useCanvasManager();
  const scale = useStore(computed(canvasManager.stage.$stageAttrs, (attrs) => attrs.scale));
  const onClick = useCallback(() => {
    const nextScale =
      SCALE_SNAPS.slice()
        .reverse()
        .find((snap) => snap < scale) ?? canvasManager.stage.config.MIN_SCALE;
    canvasManager.stage.setScale(Math.max(nextScale, canvasManager.stage.config.MIN_SCALE));
  }, [canvasManager.stage, scale]);

  return (
    <IconButton
      onClick={onClick}
      icon={<PiMagnifyingGlassMinusBold />}
      aria-label="Zoom out"
      variant="link"
      alignSelf="stretch"
      isDisabled={scale <= canvasManager.stage.config.MIN_SCALE}
    />
  );
};

const ZoomInButton = () => {
  const canvasManager = useCanvasManager();
  const scale = useStore(computed(canvasManager.stage.$stageAttrs, (attrs) => attrs.scale));
  const onClick = useCallback(() => {
    const nextScale = SCALE_SNAPS.find((snap) => snap > scale) ?? canvasManager.stage.config.MAX_SCALE;
    canvasManager.stage.setScale(Math.min(nextScale, canvasManager.stage.config.MAX_SCALE));
  }, [canvasManager.stage, scale]);

  return (
    <IconButton
      onClick={onClick}
      icon={<PiMagnifyingGlassPlusBold />}
      aria-label="Zoom out"
      variant="link"
      alignSelf="stretch"
      isDisabled={scale >= canvasManager.stage.config.MAX_SCALE}
    />
  );
};
