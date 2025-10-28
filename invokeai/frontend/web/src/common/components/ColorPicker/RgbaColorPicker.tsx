import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Button, CompositeNumberInput, Flex, FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { RGBA_COLOR_SWATCHES } from 'common/components/ColorPicker/swatches';
import { hexToRGBA, rgbaColorToString, rgbaToHex } from 'common/util/colorCodeTransformers';
import type { ChangeEvent, CSSProperties } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { RgbaColorPicker as ColorfulRgbaColorPicker } from 'react-colorful';
import type { RgbaColor } from 'react-colorful/dist/types';
import { useTranslation } from 'react-i18next';

type Props = {
  color: RgbaColor;
  onChange: (color: RgbaColor) => void;
  withNumberInput?: boolean;
  withSwatches?: boolean;
};

const colorPickerPointerStyles: NonNullable<ChakraProps['sx']> = {
  width: 6,
  height: 6,
  borderColor: 'base.100',
};

const sx: ChakraProps['sx'] = {
  '.react-colorful__hue-pointer': colorPickerPointerStyles,
  '.react-colorful__saturation-pointer': colorPickerPointerStyles,
  '.react-colorful__alpha-pointer': colorPickerPointerStyles,
  gap: 4,
  flexDir: 'column',
};

const colorPickerStyles: CSSProperties = { width: '100%' };

const numberInputWidth: ChakraProps['w'] = '3.5rem';

const RgbaColorPicker = (props: Props) => {
  const { color, onChange, withNumberInput = false, withSwatches = false } = props;
  const { t } = useTranslation();
  const handleChangeR = useCallback((r: number) => onChange({ ...color, r }), [color, onChange]);
  const handleChangeG = useCallback((g: number) => onChange({ ...color, g }), [color, onChange]);
  const handleChangeB = useCallback((b: number) => onChange({ ...color, b }), [color, onChange]);
  const handleChangeA = useCallback((a: number) => onChange({ ...color, a }), [color, onChange]);
  const [mode, setMode] = useState<'rgb' | 'hex'>('rgb');
  const [hex, setHex] = useState<string>(rgbaToHex(color, true));
  useEffect(() => {
    setHex(rgbaToHex(color, true));
  }, [color]);
  const onToggleMode = useCallback(() => setMode((m) => (m === 'rgb' ? 'hex' : 'rgb')), []);
  const onChangeHex = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      let value = e.target.value.trim();
      if (!value.startsWith('#')) {
        value = `#${value}`;
      }
      const cleaned = value.replace(/[^#0-9a-fA-F]/g, '').slice(0, 9);
      setHex(cleaned);
      const hexBody = cleaned.replace('#', '');
      if (hexBody.length === 6 || hexBody.length === 8) {
        const a = hexBody.length === 8 ? parseInt(hexBody.slice(6, 8), 16) / 255 : color.a;
        const next = hexToRGBA(hexBody.slice(0, 6).padEnd(6, '0'), a);
        onChange(next);
      }
    },
    [color.a, onChange]
  );
  const onChangeAlpha = useCallback(
    (a: number) => {
      const next = { ...color, a: Math.max(0, Math.min(1, a)) };
      onChange(next);
      setHex(rgbaToHex(next, true));
    },
    [color, onChange]
  );
  return (
    <Flex sx={sx}>
      <ColorfulRgbaColorPicker color={color} onChange={onChange} style={colorPickerStyles} />
      {withNumberInput &&
        (mode === 'rgb' ? (
          <Flex gap={2} alignItems="end">
            <Button
              size="xs"
              variant="ghost"
              px={3}
              minW="unset"
              h={10}
              whiteSpace="nowrap"
              onClick={onToggleMode}
              aria-label="Toggle RGB/HEX"
            >
              RGB
            </Button>
            <FormControl gap={0}>
              <FormLabel>{t('common.red')[0]}</FormLabel>
              <CompositeNumberInput
                value={color.r}
                onChange={handleChangeR}
                min={0}
                max={255}
                step={1}
                w={numberInputWidth}
              />
            </FormControl>
            <FormControl gap={0}>
              <FormLabel>{t('common.green')[0]}</FormLabel>
              <CompositeNumberInput
                value={color.g}
                onChange={handleChangeG}
                min={0}
                max={255}
                step={1}
                w={numberInputWidth}
              />
            </FormControl>
            <FormControl gap={0}>
              <FormLabel>{t('common.blue')[0]}</FormLabel>
              <CompositeNumberInput
                value={color.b}
                onChange={handleChangeB}
                min={0}
                max={255}
                step={1}
                w={numberInputWidth}
              />
            </FormControl>
            <FormControl gap={0}>
              <FormLabel>{t('common.alpha')[0]}</FormLabel>
              <CompositeNumberInput
                value={color.a}
                onChange={handleChangeA}
                step={0.1}
                min={0}
                max={1}
                w={numberInputWidth}
              />
            </FormControl>
          </Flex>
        ) : (
          <Flex gap={2} alignItems="end">
            <Button
              size="xs"
              variant="ghost"
              px={3}
              minW="unset"
              h={10}
              whiteSpace="nowrap"
              onClick={onToggleMode}
              aria-label="Toggle RGB/HEX"
            >
              HEX
            </Button>
            <FormControl gap={0}>
              <FormLabel>{t('common.hex', { defaultValue: 'Hex' })}</FormLabel>
              <Input value={hex} onChange={onChangeHex} placeholder="#RRGGBB or #RRGGBBAA" w="10rem" />
            </FormControl>
            <FormControl gap={0}>
              <FormLabel>{t('common.alpha')[0]}</FormLabel>
              <CompositeNumberInput
                value={color.a}
                onChange={onChangeAlpha}
                step={0.01}
                min={0}
                max={1}
                w={numberInputWidth}
              />
            </FormControl>
          </Flex>
        ))}
      {withSwatches && (
        <Flex gap={2} justifyContent="space-between">
          {RGBA_COLOR_SWATCHES.map((color, i) => (
            <ColorSwatch key={i} color={color} onChange={onChange} />
          ))}
        </Flex>
      )}
    </Flex>
  );
};

export default memo(RgbaColorPicker);

const ColorSwatch = ({ color, onChange }: { color: RgbaColor; onChange: (color: RgbaColor) => void }) => {
  const onClick = useCallback(() => {
    onChange(color);
  }, [color, onChange]);
  return <Box role="button" onClick={onClick} h={8} w={8} bg={rgbaColorToString(color)} borderRadius="base" />;
};
