import {
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import {
  selectCanvasSettingsSlice,
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { FocusEvent, KeyboardEvent, PointerEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

import { useToolIsSelected } from './hooks';

const formatPx = (v: number | string) => `${v} px`;

function mapSliderValueToRawValue(value: number) {
  if (value <= 40) {
    // 0 to 40 on the slider -> 1px to 50px
    return 1 + (49 * value) / 40;
  } else if (value <= 70) {
    // 40 to 70 on the slider -> 50px to 200px
    return 50 + (150 * (value - 40)) / 30;
  } else {
    // 70 to 100 on the slider -> 200px to 600px
    return 200 + (400 * (value - 70)) / 30;
  }
}

function mapRawValueToSliderValue(value: number) {
  if (value <= 50) {
    // 1px to 50px -> 0 to 40 on the slider
    return ((value - 1) * 40) / 49;
  } else if (value <= 200) {
    // 50px to 200px -> 40 to 70 on the slider
    return 40 + ((value - 50) * 30) / 150;
  } else {
    // 200px to 600px -> 70 to 100 on the slider
    return 70 + ((value - 200) * 30) / 400;
  }
}

function formatSliderValue(value: number) {
  return `${String(mapSliderValueToRawValue(value))} px`;
}

function formatCompactSliderValue(value: number) {
  return String(mapSliderValueToRawValue(value));
}

const marks = [
  mapRawValueToSliderValue(1),
  mapRawValueToSliderValue(50),
  mapRawValueToSliderValue(200),
  mapRawValueToSliderValue(600),
];

const sliderDefaultValue = mapRawValueToSliderValue(50);

const SLIDER_VS_DROPDOWN_CONTAINER_WIDTH_THRESHOLD = 280;
const DEFAULT_TOOL_WIDTH = 50;
const parseInputValue = (value: string) => Number.parseFloat(value);
const getInputValueFromEvent = (
  event?: Pick<FocusEvent<HTMLElement> | KeyboardEvent<HTMLElement>, 'target' | 'currentTarget'>
) => {
  const target = event?.target as HTMLInputElement | null;
  if (target?.tagName === 'INPUT') {
    return { input: target, parsed: parseInputValue(target.value) };
  }
  const currentTarget = event?.currentTarget as HTMLElement | null;
  const input = currentTarget?.querySelector('input') ?? null;
  return { input, parsed: input ? parseInputValue(input.value) : NaN };
};

interface ToolWidthPickerComponentProps {
  ariaLabel?: string;
  isCompact?: boolean;
  localValue: number;
  onChangeSlider: (value: number) => void;
  onChangeInput: (value: number) => void;
  onBlur: (event?: FocusEvent<HTMLElement>) => void;
  onKeyDown: (value: KeyboardEvent<HTMLInputElement>) => void;
  onPointerDownCapture: (value: PointerEvent<HTMLDivElement>) => void;
  onPointerUpCapture: (value: PointerEvent<HTMLDivElement>) => void;
}

const DropDownToolWidthPickerComponent = memo(
  ({
    localValue,
    ariaLabel,
    isCompact,
    onChangeSlider,
    onChangeInput,
    onKeyDown,
    onPointerDownCapture,
    onPointerUpCapture,
    onBlur,
  }: ToolWidthPickerComponentProps) => {
    const { t } = useTranslation();
    const onChangeNumberInput = useCallback(
      (valueAsString: string, valueAsNumber: number) => {
        onChangeInput(valueAsNumber);
      },
      [onChangeInput]
    );

    return (
      <Popover>
        <FormControl w="min-content" gap={2} overflow="hidden">
          <PopoverAnchor>
            <NumberInput
              size={isCompact ? 'sm' : undefined}
              variant="outline"
              display="flex"
              alignItems="center"
              min={1}
              max={600}
              value={localValue}
              onChange={onChangeNumberInput}
              onBlur={onBlur}
              w={isCompact ? 72 : 76}
              format={formatPx}
              defaultValue={50}
              onKeyDown={onKeyDown}
              onPointerDownCapture={onPointerDownCapture}
              onPointerUpCapture={onPointerUpCapture}
              clampValueOnBlur={false}
            >
              <NumberInputField _focusVisible={{ zIndex: 0 }} aria-label={ariaLabel} title="" paddingInlineEnd={7} />
              <PopoverTrigger>
                <IconButton
                  aria-label={t('common.openSlider')}
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
        <Portal>
          <PopoverContent w={200} pt={0} pb={2} px={4}>
            <PopoverArrow />
            <PopoverBody>
              <CompositeSlider
                min={0}
                max={100}
                value={mapRawValueToSliderValue(localValue)}
                onChange={onChangeSlider}
                defaultValue={sliderDefaultValue}
                marks={marks}
                formatValue={formatSliderValue}
                alwaysShowMarks
              />
            </PopoverBody>
          </PopoverContent>
        </Portal>
      </Popover>
    );
  }
);
DropDownToolWidthPickerComponent.displayName = 'DropDownToolWidthPickerComponent';

const SliderToolWidthPickerComponent = memo(
  ({
    localValue,
    ariaLabel,
    isCompact,
    onChangeSlider,
    onChangeInput,
    onKeyDown,
    onPointerDownCapture,
    onPointerUpCapture,
    onBlur,
  }: ToolWidthPickerComponentProps) => {
    return (
      <Flex w={isCompact ? 'full' : SLIDER_VS_DROPDOWN_CONTAINER_WIDTH_THRESHOLD} minW={0} gap={isCompact ? 2 : 4}>
        <CompositeSlider
          w={isCompact ? undefined : 200}
          minW={0}
          flex={isCompact ? 1 : undefined}
          h="unset"
          min={0}
          max={100}
          value={mapRawValueToSliderValue(localValue)}
          onChange={onChangeSlider}
          defaultValue={sliderDefaultValue}
          marks={marks}
          formatValue={isCompact ? formatCompactSliderValue : formatSliderValue}
          alwaysShowMarks
        />
        <CompositeNumberInput
          aria-label={ariaLabel}
          w={isCompact ? 20 : 28}
          flexShrink={0}
          variant="outline"
          min={1}
          max={600}
          value={localValue}
          onChange={onChangeInput}
          onBlur={onBlur}
          onKeyDown={onKeyDown}
          onPointerDownCapture={onPointerDownCapture}
          onPointerUpCapture={onPointerUpCapture}
          format={formatPx}
          defaultValue={50}
        />
      </Flex>
    );
  }
);
SliderToolWidthPickerComponent.displayName = 'SliderToolWidthPickerComponent';

const selectBrushWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.brushWidth);
const selectEraserWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.eraserWidth);

type ToolWidthPickerProps = {
  ariaLabel?: string;
  mode?: 'tool' | 'trace';
};

type ToolWidthHotkeysProps = {
  decrement: () => void;
  increment: () => void;
  isEnabled: boolean;
};

const ToolWidthHotkeys = memo(({ decrement, increment, isEnabled }: ToolWidthHotkeysProps) => {
  useRegisteredHotkeys({
    id: 'decrementToolWidth',
    category: 'canvas',
    callback: decrement,
    options: { enabled: isEnabled },
    dependencies: [decrement, isEnabled],
  });
  useRegisteredHotkeys({
    id: 'incrementToolWidth',
    category: 'canvas',
    callback: increment,
    options: { enabled: isEnabled },
    dependencies: [increment, isEnabled],
  });

  return null;
});

ToolWidthHotkeys.displayName = 'ToolWidthHotkeys';

export const ToolWidthPicker = memo(({ ariaLabel, mode = 'tool' }: ToolWidthPickerProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const dispatch = useAppDispatch();
  const isBrushSelected = useToolIsSelected('brush');
  const isEraserSelected = useToolIsSelected('eraser');
  const usesBrushWidth = isBrushSelected || mode === 'trace';
  const usesEraserWidth = isEraserSelected;
  const brushWidth = useAppSelector(selectBrushWidth);
  const eraserWidth = useAppSelector(selectEraserWidth);
  const width = useMemo(() => {
    if (usesBrushWidth) {
      return brushWidth;
    }
    if (usesEraserWidth) {
      return eraserWidth;
    }
    return 0;
  }, [usesBrushWidth, usesEraserWidth, brushWidth, eraserWidth]);
  const [localValue, setLocalValue] = useState(width);
  const [componentType, setComponentType] = useState<'slider' | 'dropdown' | null>(mode === 'trace' ? 'slider' : null);
  const isTypingRef = useRef(false);
  const inputPollRef = useRef<number | null>(null);

  useEffect(() => {
    if (mode === 'trace') {
      setComponentType('slider');
      return;
    }
    const el = ref.current;
    if (!el) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      for (let entry of entries) {
        if (entry.contentRect.width > SLIDER_VS_DROPDOWN_CONTAINER_WIDTH_THRESHOLD) {
          setComponentType('slider');
        } else {
          setComponentType('dropdown');
        }
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
    };
  }, [mode]);

  const onValueChange = useCallback(
    (value: number) => {
      if (usesBrushWidth) {
        dispatch(settingsBrushWidthChanged(value));
      } else if (usesEraserWidth) {
        dispatch(settingsEraserWidthChanged(value));
      }
    },
    [usesBrushWidth, usesEraserWidth, dispatch]
  );

  const onChange = useCallback(
    (value: number) => {
      onValueChange(clamp(Math.round(value), 1, 600));
    },
    [onValueChange]
  );

  const syncFromInputElement = useCallback(
    (input: HTMLInputElement | null) => {
      if (!input) {
        return;
      }
      const parsed = parseInputValue(input.value);
      if (Number.isNaN(parsed)) {
        return;
      }
      setLocalValue(parsed);
      onChange(parsed);
    },
    [onChange]
  );

  const stopPollingInput = useCallback(() => {
    if (inputPollRef.current !== null) {
      window.clearInterval(inputPollRef.current);
      inputPollRef.current = null;
    }
  }, []);

  const startPollingInput = useCallback(
    (container: HTMLElement | null) => {
      stopPollingInput();
      if (!container) {
        return;
      }
      inputPollRef.current = window.setInterval(() => {
        const input = container.querySelector('input');
        if (!input) {
          return;
        }
        const parsed = parseInputValue(input.value);
        if (Number.isNaN(parsed)) {
          return;
        }
        setLocalValue(parsed);
        if (!isTypingRef.current) {
          onChange(parsed);
        }
      }, 50);
    },
    [onChange, stopPollingInput]
  );

  const commitValue = useCallback(
    (value: number) => {
      if (isNaN(Number(value))) {
        onChange(DEFAULT_TOOL_WIDTH);
        setLocalValue(DEFAULT_TOOL_WIDTH);
      } else {
        onChange(value);
        setLocalValue(value);
      }
    },
    [onChange]
  );

  const increment = useCallback(() => {
    let newWidth = Math.round(width * 1.15);
    if (newWidth === width) {
      newWidth += 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  const decrement = useCallback(() => {
    let newWidth = Math.round(width * 0.85);
    if (newWidth === width) {
      newWidth -= 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  const onChangeSlider = useCallback(
    (value: number) => {
      onChange(mapSliderValueToRawValue(value));
    },
    [onChange]
  );

  const onChangeInput = useCallback(
    (value: number) => {
      setLocalValue(value);
      if (!isNaN(value) && !isTypingRef.current) {
        onChange(value);
      }
    },
    [onChange]
  );

  const onBlur = useCallback(
    (event?: FocusEvent<HTMLElement>) => {
      const { parsed } = getInputValueFromEvent(event);
      commitValue(Number.isNaN(parsed) ? localValue : parsed);
      isTypingRef.current = false;
    },
    [commitValue, localValue]
  );

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        const { parsed } = getInputValueFromEvent(e);
        commitValue(Number.isNaN(parsed) ? localValue : parsed);
        isTypingRef.current = false;
        return;
      }
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        isTypingRef.current = false;
        const { input } = getInputValueFromEvent(e);
        window.requestAnimationFrame(() => {
          syncFromInputElement(input);
        });
        return;
      }
      if (e.key === 'Backspace' || e.key === 'Delete' || e.key.length === 1) {
        isTypingRef.current = true;
      }
    },
    [commitValue, localValue, syncFromInputElement]
  );

  const onPointerDownCapture = useCallback(
    (_e: PointerEvent<HTMLDivElement>) => {
      isTypingRef.current = false;
      const target = _e.target as HTMLElement | null;
      if (target && target.tagName !== 'INPUT') {
        startPollingInput(_e.currentTarget);
      } else {
        stopPollingInput();
      }
    },
    [startPollingInput, stopPollingInput]
  );

  const onPointerUpCapture = useCallback(() => {
    stopPollingInput();
  }, [stopPollingInput]);

  useEffect(() => {
    setLocalValue(width);
  }, [width]);

  useEffect(() => {
    return () => {
      stopPollingInput();
    };
  }, [stopPollingInput]);

  return (
    <Flex ref={ref} alignItems="center" flexGrow={1} flexShrink={1} minW={0}>
      {mode === 'tool' && (
        <ToolWidthHotkeys decrement={decrement} increment={increment} isEnabled={isBrushSelected || isEraserSelected} />
      )}
      {componentType === 'slider' && (
        <SliderToolWidthPickerComponent
          ariaLabel={ariaLabel}
          isCompact={mode === 'trace'}
          localValue={localValue}
          onChangeSlider={onChangeSlider}
          onChangeInput={onChangeInput}
          onBlur={onBlur}
          onKeyDown={onKeyDown}
          onPointerDownCapture={onPointerDownCapture}
          onPointerUpCapture={onPointerUpCapture}
        />
      )}
      {componentType === 'dropdown' && (
        <DropDownToolWidthPickerComponent
          ariaLabel={ariaLabel}
          isCompact={mode === 'trace'}
          localValue={localValue}
          onChangeSlider={onChangeSlider}
          onChangeInput={onChangeInput}
          onBlur={onBlur}
          onKeyDown={onKeyDown}
          onPointerDownCapture={onPointerDownCapture}
          onPointerUpCapture={onPointerUpCapture}
        />
      )}
    </Flex>
  );
});

ToolWidthPicker.displayName = 'ToolWidthPicker';
