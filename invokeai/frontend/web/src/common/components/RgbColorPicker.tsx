import type { ChakraProps } from '@invoke-ai/ui-library';
import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { RgbColorPicker as ColorfulRgbColorPicker } from 'react-colorful';
import type { ColorPickerBaseProps, RgbColor } from 'react-colorful/dist/types';
import { useTranslation } from 'react-i18next';

type RgbColorPickerProps = ColorPickerBaseProps<RgbColor> & {
  withNumberInput?: boolean;
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
  gap: 5,
  flexDir: 'column',
};

const colorPickerStyles: CSSProperties = { width: '100%' };

const numberInputWidth: ChakraProps['w'] = '3.5rem';

const RgbColorPicker = (props: RgbColorPickerProps) => {
  const { color, onChange, withNumberInput, ...rest } = props;
  const { t } = useTranslation();
  const handleChangeR = useCallback((r: number) => onChange({ ...color, r }), [color, onChange]);
  const handleChangeG = useCallback((g: number) => onChange({ ...color, g }), [color, onChange]);
  const handleChangeB = useCallback((b: number) => onChange({ ...color, b }), [color, onChange]);
  return (
    <Flex sx={sx}>
      <ColorfulRgbColorPicker color={color} onChange={onChange} style={colorPickerStyles} {...rest} />
      {withNumberInput && (
        <Flex gap={5}>
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
        </Flex>
      )}
    </Flex>
  );
};

export default memo(RgbColorPicker);
