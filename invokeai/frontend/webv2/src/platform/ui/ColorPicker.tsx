import type { Color, ColorPickerValueChangeDetails } from '@chakra-ui/react';
import type { ChangeEvent, ComponentProps } from 'react';

import {
  ColorPicker as ChakraColorPicker,
  getColorChannels,
  HStack,
  Icon,
  Input,
  parseColor,
  Portal,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Pipette } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ColorPickerFormat } from './colorPickerStore';

import { IconButton } from './Button';
import { normalizeHex } from './color';
import {
  COLOR_PICKER_FORMATS,
  DEFAULT_COLOR_SWATCHES,
  recordRecentColor,
  setColorPickerFormat,
  useColorPickerFormat,
  useRecentColors,
} from './colorPickerStore';
import { shouldSyncExternalColor } from './colorPickerSync';

export type ColorPickerSize = NonNullable<ComponentProps<typeof ChakraColorPicker.Root>['size']>;

export type { ColorPickerFormat };

/**
 * Machine formats behind each presented format. `hex` and `rgb` share `rgba`,
 * so switching between them changes only which inputs render — never the value.
 */
const MACHINE_FORMAT: Record<ColorPickerFormat, 'rgba' | 'hsla' | 'hsba'> = {
  hex: 'rgba',
  hsb: 'hsba',
  hsl: 'hsla',
  rgb: 'rgba',
};

/**
 * The checkerboard behind transparent colors. This is a CSS *length* (it drives
 * `background-size` on a conic gradient), not a size token — passing `"xs"`
 * yields an invalid `background-size` and the gradient renders once, stretched,
 * instead of tiling. Chakra's own default is `0.6rem`.
 */
const TRANSPARENCY_CHECK_SIZE = '0.5rem';

/**
 * Swatches wrap into as many even columns as the popover actually fits.
 *
 * `auto-fill` rather than a fixed `repeat(10, …)`: the swatch holds a fixed
 * `--swatch-size` from the size variant, so a fixed column count overflows the
 * popover the moment the swatch, the gap, or the content width changes — which
 * is exactly what happened at ten columns in a 224px popover. Deriving the
 * count from the available width instead makes that arithmetic unrepresentable.
 */
const SWATCH_GROUP_CSS = {
  display: 'grid',
  gap: '1',
  gridTemplateColumns: 'repeat(auto-fill, minmax(var(--swatch-size, 1.25rem), 1fr))',
};

const CHANNEL_ROW_CSS = { '& input': { minW: '0' } };
const SLIDER_STACK_CSS = { minW: '0' };

export interface ColorPickerProps {
  /** Accessible label for the swatch trigger button. */
  'aria-label': string;
  disabled?: boolean;
  size?: ColorPickerSize;
  /**
   * The swatch row: `true` (default) shows the workbench palette followed by
   * recently committed colors, `false` hides it, and an array supplies an
   * explicit palette instead.
   */
  swatches?: boolean | readonly string[];
  /** The current color, as a `#rrggbb` (or `#rrggbbaa` under `withAlpha`) string. */
  value: string;
  /**
   * Accept and emit `#rrggbbaa`, and show the alpha slider, alpha channel
   * input, and a transparency checkerboard behind the trigger swatch.
   */
  withAlpha?: boolean;
  /**
   * Offer the browser's screen eyedropper. Chromium-only — the trigger is
   * omitted entirely where `EyeDropper` is unavailable rather than rendering a
   * button that does nothing. Ignored when `onSampleColor` is supplied.
   */
  withEyeDropper?: boolean;
  /** Show the current value as text beside the trigger swatch. */
  withValueText?: boolean;
  /** Called as the user drags the area/sliders or edits a channel input. */
  onValueChange: (value: string) => void;
  /**
   * Called with the final value when an interaction ends (pointer up on the
   * area/slider, a channel input committing, or a swatch being picked).
   * Consumers that record undo history use this to collapse a whole drag into
   * one entry, mirroring the `Slider` `onValueChangeEnd` pattern.
   */
  onValueChangeEnd?: (value: string) => void;
  /**
   * Sample a color from somewhere the picker cannot reach itself — the canvas
   * passes a sampler backed by its own eyedropper tool, which reads the
   * composited document rather than the screen. Replaces the browser
   * eyedropper when present. Resolves `null` when the user cancels.
   */
  onSampleColor?: () => Promise<string | null>;
}

/**
 * The colour channels a format exposes. Alpha is always dropped — it gets its
 * own percentage field ({@link AlphaInput}) rather than Zag's unit-float one,
 * and it belongs at the end of the row regardless of which format is showing.
 */
const useChannels = (format: ColorPickerFormat) =>
  useMemo(() => getColorChannels(MACHINE_FORMAT[format]).filter((channel) => channel !== 'alpha'), [format]);

/**
 * Cycles HEX → RGB → HSL → HSB.
 *
 * A dropdown would name all four up front, but it has to portal out of the
 * picker's own popover, and opening it reads as an outside interaction that
 * dismisses the picker. Ark's built-in `FormatTrigger` cycles for the same
 * reason; this keeps that behavior on workbench chrome.
 */
const FormatTrigger = ({
  format,
  onFormatChange,
}: {
  format: ColorPickerFormat;
  onFormatChange: (format: ColorPickerFormat) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    const index = COLOR_PICKER_FORMATS.indexOf(format);
    onFormatChange(COLOR_PICKER_FORMATS[(index + 1) % COLOR_PICKER_FORMATS.length]);
  }, [format, onFormatChange]);

  return (
    // `subtle` rather than `ghost`: a bare label beside two filled inputs reads
    // as a caption, not something you can click to change the units.
    <IconButton
      aria-label={t('common.colorPicker.format')}
      flexShrink="0"
      fontSize="2xs"
      fontWeight="medium"
      size="xs"
      variant="subtle"
      w="10"
      onClick={handleClick}
    >
      {format.toUpperCase()}
    </IconButton>
  );
};

/**
 * A swatch that also reports a commit. Zag's `SwatchTrigger` sets the value and
 * fires `onValueChange`, but never `onValueChangeEnd` — without this, every
 * consumer that records undo history on commit (mask fills, shapes, gradients)
 * would silently drop swatch picks from their history.
 *
 * The commit is *armed* here rather than emitted: Zag's own value change runs
 * after this handler, so emitting directly would put the commit before the
 * change and leave undo entries recording the pre-swatch color.
 */
const CommittingSwatch = ({ value, onArmCommit }: { value: string; onArmCommit: () => void }) => (
  <ChakraColorPicker.SwatchTrigger value={value} onClick={onArmCommit}>
    <ChakraColorPicker.Swatch value={value}>
      <ChakraColorPicker.SwatchIndicator boxSize="2.5" />
    </ChakraColorPicker.Swatch>
  </ChakraColorPicker.SwatchTrigger>
);

/**
 * Opacity as a whole percentage.
 *
 * Zag's own alpha `ChannelInput` renders the raw unit float — a field reading
 * `0.501` next to a hex triplet is noise, and nobody types opacity that way.
 * This edits the same channel in the units the rest of the app uses (layer
 * opacity, brush opacity) and commits on blur so a drag-free edit is one entry.
 */
const AlphaInput = ({
  alpha,
  onAlphaChange,
  onAlphaCommit,
}: {
  alpha: number;
  onAlphaChange: (alpha: number) => void;
  onAlphaCommit: () => void;
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<string | null>(null);
  const handleChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const raw = event.currentTarget.value;
      setDraft(raw);
      const parsed = Number.parseInt(raw, 10);
      if (Number.isFinite(parsed)) {
        onAlphaChange(Math.min(100, Math.max(0, parsed)) / 100);
      }
    },
    [onAlphaChange]
  );
  const handleBlur = useCallback(() => {
    setDraft(null);
    onAlphaCommit();
  }, [onAlphaCommit]);

  return (
    <Input
      aria-label={t('common.colorPicker.opacity')}
      className="colorPicker__channelInput"
      flexShrink="0"
      inputMode="numeric"
      size="xs"
      value={draft ?? String(Math.round(alpha * 100))}
      w="12"
      onBlur={handleBlur}
      onChange={handleChange}
    />
  );
};

/**
 * Workbench color picker: a compact swatch trigger that opens a popover with a
 * saturation/value area, hue and alpha sliders, an eyedropper, switchable
 * HEX/RGB/HSL/HSB channel inputs, and a swatch row of the workbench palette
 * plus recently committed colors. Composed from Chakra v3's `ColorPicker.*`
 * parts; chrome comes from the `colorPicker` slot-recipe override in
 * `theme/recipes.ts`.
 *
 * Internally, the picker's controlled value is kept as a full Chakra/Zag
 * `Color` (which preserves hue/saturation independent of RGB), not a hex
 * string — see `shouldSyncExternalColor` for why. The external API stays
 * string based; hex is only produced at the emit boundary.
 */
export const ColorPicker = ({
  'aria-label': ariaLabel,
  disabled,
  onSampleColor,
  onValueChange,
  onValueChangeEnd,
  size,
  swatches = true,
  value,
  withAlpha = false,
  withEyeDropper = true,
  withValueText = false,
}: ColorPickerProps) => {
  const { t } = useTranslation();
  const format = useColorPickerFormat();
  const recents = useRecentColors();
  const channels = useChannels(format);

  // Zag emits uppercase hex; the workbench stores and compares lowercase.
  const toEmitted = useCallback(
    (color: Color) => color.toString(withAlpha ? 'hexa' : 'hex').toLowerCase(),
    [withAlpha]
  );

  const isSwatchCommitArmed = useRef(false);
  const [isOpen, setIsOpen] = useState(false);
  const [previousExternalValue, setPreviousExternalValue] = useState(value);
  const [color, setColor] = useState<Color>(() => parseColor(value));
  const [lastEmittedValue, setLastEmittedValue] = useState(() => toEmitted(color));

  // Sync external -> internal only when the prop genuinely changed to
  // something other than what we last emitted (see `shouldSyncExternalColor`).
  if (value !== previousExternalValue) {
    setPreviousExternalValue(value);
    if (shouldSyncExternalColor(value, previousExternalValue, lastEmittedValue)) {
      setColor(parseColor(value));
    }
  }

  const emit = useCallback(
    (next: Color, isEnd: boolean) => {
      setColor(next);
      const emitted = toEmitted(next);
      setLastEmittedValue(emitted);
      onValueChange(emitted);
      if (isEnd) {
        recordRecentColor(emitted);
        onValueChangeEnd?.(emitted);
      }
    },
    [onValueChange, onValueChangeEnd, toEmitted]
  );

  const handleValueChange = useCallback(
    (details: ColorPickerValueChangeDetails) => {
      // A swatch pick is both a change and a commit; Zag only reports the
      // change, so `CommittingSwatch` arms the commit and it lands here in the
      // right order.
      const isCommit = isSwatchCommitArmed.current;
      isSwatchCommitArmed.current = false;
      emit(details.value, isCommit);
    },
    [emit]
  );

  const armSwatchCommit = useCallback(() => {
    isSwatchCommitArmed.current = true;
  }, []);

  const handleValueChangeEnd = useCallback(
    (details: ColorPickerValueChangeDetails) => {
      const emitted = toEmitted(details.value);
      setLastEmittedValue(emitted);
      recordRecentColor(emitted);
      onValueChangeEnd?.(emitted);
    },
    [onValueChangeEnd, toEmitted]
  );

  const alpha = withAlpha ? color.getChannelValue('alpha') : 1;

  const handleAlphaChange = useCallback(
    (next: number) => emit(color.withChannelValue('alpha', next) as Color, false),
    [color, emit]
  );

  const handleAlphaCommit = useCallback(() => {
    recordRecentColor(lastEmittedValue);
    onValueChangeEnd?.(lastEmittedValue);
  }, [lastEmittedValue, onValueChangeEnd]);

  const handleOpenChange = useCallback((details: { open: boolean }) => setIsOpen(details.open), []);

  const handleSampleColor = useCallback(async () => {
    if (!onSampleColor) {
      return;
    }
    // Get out of the way so the user can see (and click) what they're sampling.
    setIsOpen(false);
    const sampled = await onSampleColor();
    if (sampled) {
      emit(parseColor(sampled), true);
    }
  }, [emit, onSampleColor]);

  const swatchList = useMemo(() => {
    if (Array.isArray(swatches)) {
      return swatches as readonly string[];
    }
    if (!swatches) {
      return [];
    }

    const seen = new Set(DEFAULT_COLOR_SWATCHES.map((entry) => normalizeHex(entry)));

    return [...DEFAULT_COLOR_SWATCHES, ...recents.filter((entry) => !seen.has(normalizeHex(entry)))];
  }, [recents, swatches]);

  // The browser eyedropper is Chromium-only; omit the trigger where it would
  // do nothing. A canvas-backed sampler always wins when one is supplied.
  const canUseScreenEyeDropper = withEyeDropper && typeof window !== 'undefined' && 'EyeDropper' in window;

  return (
    <ChakraColorPicker.Root
      disabled={disabled}
      format={MACHINE_FORMAT[format]}
      lazyMount
      open={isOpen}
      size={size}
      unmountOnExit
      value={color}
      onOpenChange={handleOpenChange}
      onValueChange={handleValueChange}
      onValueChangeEnd={handleValueChangeEnd}
    >
      <ChakraColorPicker.Control>
        <ChakraColorPicker.Trigger aria-label={ariaLabel} data-fit-content>
          {withAlpha ? <ChakraColorPicker.TransparencyGrid size={TRANSPARENCY_CHECK_SIZE} /> : null}
          <ChakraColorPicker.ValueSwatch />
        </ChakraColorPicker.Trigger>
        {withValueText ? (
          <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
            {lastEmittedValue}
          </Text>
        ) : null}
      </ChakraColorPicker.Control>
      <Portal>
        <ChakraColorPicker.Positioner>
          <ChakraColorPicker.Content>
            <ChakraColorPicker.Area>
              <ChakraColorPicker.AreaBackground />
              <ChakraColorPicker.AreaThumb />
            </ChakraColorPicker.Area>

            <HStack align="center" gap="2">
              {onSampleColor ? (
                <IconButton
                  aria-label={t('common.colorPicker.sampleFromCanvas')}
                  size="xs"
                  variant="ghost"
                  onClick={handleSampleColor}
                >
                  <Icon as={Pipette} boxSize="4" />
                </IconButton>
              ) : canUseScreenEyeDropper ? (
                // `EyeDropperTrigger` is a bare Ark part, not an IconButton —
                // rendered directly it has no icon and drops `variant`/`size`
                // onto the DOM as raw attributes, collapsing to a 0x0 button.
                // `asChild` puts the machine's behavior on workbench chrome, so
                // both eyedropper branches look identical.
                <ChakraColorPicker.EyeDropperTrigger asChild>
                  <IconButton aria-label={t('common.colorPicker.sampleFromScreen')} size="xs" variant="ghost">
                    <Icon as={Pipette} boxSize="4" />
                  </IconButton>
                </ChakraColorPicker.EyeDropperTrigger>
              ) : null}
              <Stack css={SLIDER_STACK_CSS} flex="1" gap="1.5">
                <ChakraColorPicker.ChannelSlider channel="hue">
                  <ChakraColorPicker.ChannelSliderTrack />
                  <ChakraColorPicker.ChannelSliderThumb />
                </ChakraColorPicker.ChannelSlider>
                {withAlpha ? (
                  <ChakraColorPicker.ChannelSlider channel="alpha">
                    <ChakraColorPicker.TransparencyGrid size={TRANSPARENCY_CHECK_SIZE} />
                    <ChakraColorPicker.ChannelSliderTrack />
                    <ChakraColorPicker.ChannelSliderThumb />
                  </ChakraColorPicker.ChannelSlider>
                ) : null}
              </Stack>
            </HStack>

            <HStack css={CHANNEL_ROW_CSS} gap="1">
              <FormatTrigger format={format} onFormatChange={setColorPickerFormat} />
              {format === 'hex' ? (
                <ChakraColorPicker.ChannelInput channel="hex" flex="1" />
              ) : (
                channels.map((channel) => <ChakraColorPicker.ChannelInput key={channel} channel={channel} flex="1" />)
              )}
              {withAlpha ? (
                <AlphaInput alpha={alpha} onAlphaChange={handleAlphaChange} onAlphaCommit={handleAlphaCommit} />
              ) : null}
            </HStack>

            {swatchList.length > 0 ? (
              <ChakraColorPicker.SwatchGroup
                borderColor="border.subtle"
                borderTopWidth="1px"
                css={SWATCH_GROUP_CSS}
                pt="2"
              >
                {swatchList.map((swatch) => (
                  <CommittingSwatch key={swatch} value={swatch} onArmCommit={armSwatchCommit} />
                ))}
              </ChakraColorPicker.SwatchGroup>
            ) : null}
          </ChakraColorPicker.Content>
        </ChakraColorPicker.Positioner>
      </Portal>
    </ChakraColorPicker.Root>
  );
};
