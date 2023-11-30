import { ChakraProps, Flex } from '@chakra-ui/react';
import { memo, useCallback } from 'react';
import { RgbaColorPicker } from 'react-colorful';
import { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types';
import IAINumberInput from './IAINumberInput';

type IAIColorPickerProps = ColorPickerBaseProps<RgbaColor> & {
  withNumberInput?: boolean;
};

const colorPickerStyles: NonNullable<ChakraProps['sx']> = {
  width: 6,
  height: 6,
  borderColor: 'base.100',
};

const sx: ChakraProps['sx'] = {
  '.react-colorful__hue-pointer': colorPickerStyles,
  '.react-colorful__saturation-pointer': colorPickerStyles,
  '.react-colorful__alpha-pointer': colorPickerStyles,
  gap: 2,
  flexDir: 'column',
};

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
        style={{ width: '100%' }}
        {...rest}
      />
      {withNumberInput && (
        <Flex>
          <IAINumberInput
            value={color.r}
            onChange={handleChangeR}
            min={0}
            max={255}
            step={1}
            label="Red"
            w={numberInputWidth}
          />
          <IAINumberInput
            value={color.g}
            onChange={handleChangeG}
            min={0}
            max={255}
            step={1}
            label="Green"
            w={numberInputWidth}
          />
          <IAINumberInput
            value={color.b}
            onChange={handleChangeB}
            min={0}
            max={255}
            step={1}
            label="Blue"
            w={numberInputWidth}
          />
          <IAINumberInput
            value={color.a}
            onChange={handleChangeA}
            step={0.1}
            min={0}
            max={1}
            label="Alpha"
            w={numberInputWidth}
            isInteger={false}
          />
        </Flex>
      )}
    </Flex>
  );
};

export default memo(IAIColorPicker);
