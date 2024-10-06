import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { RGBA_COLOR_SWATCHES } from 'common/components/ColorPicker/swatches';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
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
  return (
    <Flex sx={sx}>
      <ColorfulRgbaColorPicker color={color} onChange={onChange} style={colorPickerStyles} />
      {withNumberInput && (
        <Flex gap={2}>
          <FormControl gap={0}>
            <FormLabel>{t('common.red')[0]}</FormLabel>
            <CompositeNumberInput
              value={color.r}
              onChange={handleChangeR}
              min={0}
              max={255}
              step={1}
              w={numberInputWidth}
              defaultValue={90}
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
              defaultValue={90}
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
              defaultValue={255}
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
              defaultValue={1}
            />
          </FormControl>
        </Flex>
      )}
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
