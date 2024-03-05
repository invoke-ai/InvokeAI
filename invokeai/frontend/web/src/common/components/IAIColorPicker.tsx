import type { ChakraProps } from '@invoke-ai/ui-library';
import { CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { RgbaColorPicker } from 'react-colorful';
import type { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types';
import { useTranslation } from 'react-i18next';

type IAIColorPickerProps = ColorPickerBaseProps<RgbaColor> & {
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

const numberInputWidth: ChakraProps['w'] = '4.2rem';

const IAIColorPicker = (props: IAIColorPickerProps) => {
  const { color, onChange, withNumberInput, ...rest } = props;
  const { t } = useTranslation();
  const handleChangeR = useCallback((r: number) => onChange({ ...color, r }), [color, onChange]);
  const handleChangeG = useCallback((g: number) => onChange({ ...color, g }), [color, onChange]);
  const handleChangeB = useCallback((b: number) => onChange({ ...color, b }), [color, onChange]);
  const handleChangeA = useCallback((a: number) => onChange({ ...color, a }), [color, onChange]);
  return (
    <Flex sx={sx}>
      <RgbaColorPicker color={color} onChange={onChange} style={colorPickerStyles} {...rest} />
      {withNumberInput && (
        <Flex gap={5}>
          <FormControl gap={0}>
            <FormLabel>{t('common.red')}</FormLabel>
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
            <FormLabel>{t('common.green')}</FormLabel>
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
            <FormLabel>{t('common.blue')}</FormLabel>
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
            <FormLabel>{t('common.alpha')}</FormLabel>
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
    </Flex>
  );
};

export default memo(IAIColorPicker);
