import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { RGB_COLOR_SWATCHES } from 'common/components/ColorPicker/swatches';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { RgbColorPicker as ColorfulRgbColorPicker } from 'react-colorful';
import type { RgbColor } from 'react-colorful/dist/types';
import { useTranslation } from 'react-i18next';

type Props = {
  color: RgbColor;
  onChange: (color: RgbColor) => void;
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

const RgbColorPicker = (props: Props) => {
  const { color, onChange, withNumberInput = false, withSwatches = false } = props;
  const { t } = useTranslation();
  const handleChangeR = useCallback((r: number) => onChange({ ...color, r }), [color, onChange]);
  const handleChangeG = useCallback((g: number) => onChange({ ...color, g }), [color, onChange]);
  const handleChangeB = useCallback((b: number) => onChange({ ...color, b }), [color, onChange]);
  return (
    <Flex sx={sx}>
      <ColorfulRgbColorPicker color={color} onChange={onChange} style={colorPickerStyles} />
      {withNumberInput && (
        <Flex gap={4}>
          <FormControl gap={0}>
            <FormLabel>{t('common.red')[0]}</FormLabel>
            <CompositeNumberInput
              flexGrow={1}
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
              flexGrow={1}
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
              flexGrow={1}
              value={color.b}
              onChange={handleChangeB}
              min={0}
              max={255}
              step={1}
              w={numberInputWidth}
              defaultValue={255}
            />
          </FormControl>
        </Flex>
      )}
      {withSwatches && (
        <Flex gap={2} justifyContent="space-between">
          {RGB_COLOR_SWATCHES.map((color, i) => (
            <ColorSwatch key={i} color={color} onChange={onChange} />
          ))}
        </Flex>
      )}
    </Flex>
  );
};

export default memo(RgbColorPicker);

const ColorSwatch = ({ color, onChange }: { color: RgbColor; onChange: (color: RgbColor) => void }) => {
  const onClick = useCallback(() => {
    onChange(color);
  }, [color, onChange]);
  return <Box role="button" onClick={onClick} h={8} w={8} bg={rgbColorToString(color)} borderRadius="base" />;
};
