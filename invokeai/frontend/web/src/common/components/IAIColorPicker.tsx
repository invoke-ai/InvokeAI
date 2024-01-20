import type { ChakraProps } from '@invoke-ai/ui';
import {
  CompositeNumberInput,
  Flex,
  FormControl,
  FormLabel,
} from '@invoke-ai/ui';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { RgbaColorPicker } from 'react-colorful';
import type {
  ColorPickerBaseProps,
  RgbaColor,
} from 'react-colorful/dist/types';

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
  gap: 2,
  flexDir: 'column',
};

const colorPickerStyles: CSSProperties = { width: '100%' };

const numberInputWidth: ChakraProps['w'] = '4.2rem';

const IAIColorPicker = (props: IAIColorPickerProps) => {
  const { color, onChange, withNumberInput, ...rest } = props;
  const handleChangeR = useCallback(
    (r: number) => onChange({ ...color, r }),
    [color, onChange]
  );
  const handleChangeG = useCallback(
    (g: number) => onChange({ ...color, g }),
    [color, onChange]
  );
  const handleChangeB = useCallback(
    (b: number) => onChange({ ...color, b }),
    [color, onChange]
  );
  const handleChangeA = useCallback(
    (a: number) => onChange({ ...color, a }),
    [color, onChange]
  );
  return (
    <Flex sx={sx}>
      <RgbaColorPicker
        color={color}
        onChange={onChange}
        style={colorPickerStyles}
        {...rest}
      />
      {withNumberInput && (
        <Flex>
          <FormControl>
            <FormLabel>Red</FormLabel>
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
          <FormControl>
            <FormLabel>Green</FormLabel>
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
          <FormControl>
            <FormLabel>Blue</FormLabel>
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
          <FormControl>
            <FormLabel>Alpha</FormLabel>
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
