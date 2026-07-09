import type { Color, ColorPickerValueChangeDetails } from '@chakra-ui/react';
import type { ComponentProps } from 'react';

import { ColorPicker as ChakraColorPicker, parseColor, Portal } from '@chakra-ui/react';
import { useCallback, useState } from 'react';

import { shouldSyncExternalColor } from './colorPickerSync';

export type ColorPickerSize = NonNullable<ComponentProps<typeof ChakraColorPicker.Root>['size']>;

export interface ColorPickerProps {
  /** The current color, as a `#rrggbb` (or any CSS-parseable) string. */
  value: string;
  /** Called with the new `#rrggbb` hex string as the user drags the area/slider or edits the hex input. */
  onValueChange: (hex: string) => void;
  /**
   * Called with the final `#rrggbb` hex when an interaction ends (pointer up on
   * the area/slider, or the hex input committing). Consumers that record undo
   * history use this to collapse a whole drag into one entry, mirroring the
   * `Slider` `onValueChangeEnd` pattern.
   */
  onValueChangeEnd?: (hex: string) => void;
  /** Accessible label for the swatch trigger button. */
  'aria-label': string;
  size?: ColorPickerSize;
}

/**
 * Workbench color picker: a compact swatch trigger that opens a popover with a
 * saturation/hue area and a hex input. Wraps Chakra v3's composed
 * `ColorPicker.*` parts (chrome comes from the built-in `colorPicker` slot
 * recipe, which already reads workbench semantic tokens) — this is the single
 * import point, kept to exactly what `BrushOptions` needs.
 *
 * Internally, the picker's controlled value is kept as a full Chakra/Zag
 * `Color` (which preserves hue/saturation independent of RGB), not a hex
 * string — see `shouldSyncExternalColor` for why. The external API stays
 * hex-string based; hex is only produced at the `onValueChange` emit boundary.
 */
export const ColorPicker = ({
  'aria-label': ariaLabel,
  onValueChange,
  onValueChangeEnd,
  size = 'xs',
  value,
}: ColorPickerProps) => {
  const [previousExternalValue, setPreviousExternalValue] = useState(value);
  const [color, setColor] = useState<Color>(() => parseColor(value));
  const [lastEmittedHex, setLastEmittedHex] = useState(() => color.toString('hex'));

  // Sync external -> internal only when the prop genuinely changed to
  // something other than what we last emitted (see `shouldSyncExternalColor`).
  if (value !== previousExternalValue) {
    setPreviousExternalValue(value);
    if (shouldSyncExternalColor(value, previousExternalValue, lastEmittedHex)) {
      setColor(parseColor(value));
    }
  }

  const handleValueChange = useCallback(
    (details: ColorPickerValueChangeDetails) => {
      setColor(details.value);
      const hex = details.value.toString('hex');
      setLastEmittedHex(hex);
      onValueChange(hex);
    },
    [onValueChange]
  );

  const handleValueChangeEnd = useCallback(
    (details: ColorPickerValueChangeDetails) => {
      onValueChangeEnd?.(details.value.toString('hex'));
    },
    [onValueChangeEnd]
  );

  return (
    <ChakraColorPicker.Root
      size={size}
      value={color}
      onValueChange={handleValueChange}
      onValueChangeEnd={handleValueChangeEnd}
    >
      <ChakraColorPicker.Control>
        <ChakraColorPicker.Trigger aria-label={ariaLabel} data-fit-content>
          <ChakraColorPicker.ValueSwatch />
        </ChakraColorPicker.Trigger>
      </ChakraColorPicker.Control>
      <Portal>
        <ChakraColorPicker.Positioner>
          <ChakraColorPicker.Content>
            <ChakraColorPicker.Area>
              <ChakraColorPicker.AreaBackground />
              <ChakraColorPicker.AreaThumb />
            </ChakraColorPicker.Area>
            <ChakraColorPicker.ChannelSlider channel="hue">
              <ChakraColorPicker.ChannelSliderTrack />
              <ChakraColorPicker.ChannelSliderThumb />
            </ChakraColorPicker.ChannelSlider>
            <ChakraColorPicker.Input />
          </ChakraColorPicker.Content>
        </ChakraColorPicker.Positioner>
      </Portal>
    </ChakraColorPicker.Root>
  );
};
