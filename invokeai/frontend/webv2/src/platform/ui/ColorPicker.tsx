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

const SWATCH_GROUP_CSS = {
  display: 'grid',
  gap: '1',
  gridTemplateColumns: 'repeat(auto-fill, minmax(var(--swatch-size, 1.25rem), 1fr))',
};

const CHANNEL_ROW_CSS = { '& input': { minW: '0' } };
const SLIDER_STACK_CSS = { minW: '0' };

export interface ColorPickerProps {
  'aria-label': string;
  disabled?: boolean;
  size?: ColorPickerSize;
  swatches?: boolean | readonly string[];
  value: string;
  withAlpha?: boolean;
  withEyeDropper?: boolean;
  withValueText?: boolean;
  onValueChange: (value: string) => void;
  onValueChangeEnd?: (value: string) => void;
  onSampleColor?: () => Promise<string | null>;
}

const useChannels = (format: ColorPickerFormat) =>
  useMemo(() => getColorChannels(MACHINE_FORMAT[format]).filter((channel) => channel !== 'alpha'), [format]);

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

// Zag does not emit onValueChangeEnd for swatches, and its value change runs after the click handler.
const CommittingSwatch = ({ value, onArmCommit }: { value: string; onArmCommit: () => void }) => (
  <ChakraColorPicker.SwatchTrigger value={value} onClick={onArmCommit}>
    <ChakraColorPicker.Swatch value={value}>
      <ChakraColorPicker.SwatchIndicator boxSize="2.5" />
    </ChakraColorPicker.Swatch>
  </ChakraColorPicker.SwatchTrigger>
);

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
