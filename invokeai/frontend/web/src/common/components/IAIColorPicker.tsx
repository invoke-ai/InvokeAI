import { Box, ChakraProps, Flex } from '@chakra-ui/react';
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

const sx = {
  '.react-colorful__hue-pointer': colorPickerStyles,
  '.react-colorful__saturation-pointer': colorPickerStyles,
  '.react-colorful__alpha-pointer': colorPickerStyles,
};

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
    <Box sx={sx}>
      <RgbaColorPicker color={color} onChange={onChange} {...rest} />
      <Box>
        {withNumberInput && (
          <IAINumberInput
            value={color.r}
            onChange={handleChangeR}
            min={0}
            max={255}
            step={1}
            label="Red"
          />
        )}
        {withNumberInput && (
          <IAINumberInput
            value={color.g}
            onChange={handleChangeG}
            min={0}
            max={255}
            step={1}
            label="Green"
          />
        )}
        {withNumberInput && (
          <IAINumberInput
            value={color.b}
            onChange={handleChangeB}
            min={0}
            max={255}
            step={1}
            label="Blue"
          />
        )}
        {withNumberInput && (
          <IAINumberInput
            value={color.a}
            onChange={handleChangeA}
            step={0.1}
            min={0}
            max={1}
            label="Alpha"
            isInteger={false}
          />
        )}
      </Box>
    </Box>
  );
};

export default memo(IAIColorPicker);
