import { chakra, ChakraProps } from '@chakra-ui/react';
import { memo } from 'react';
import { RgbaColorPicker } from 'react-colorful';
import { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types';

type IAIColorPickerProps = Omit<ColorPickerBaseProps<RgbaColor>, 'color'> &
  ChakraProps & {
    pickerColor: RgbaColor;
    styleClass?: string;
  };

const ChakraRgbaColorPicker = chakra(RgbaColorPicker, {
  baseStyle: { paddingInline: 4 },
  shouldForwardProp: (prop) => !['pickerColor'].includes(prop),
});

const colorPickerStyles: NonNullable<ChakraProps['sx']> = {
  width: 6,
  height: 6,
  borderColor: 'base.100',
};

const IAIColorPicker = (props: IAIColorPickerProps) => {
  const { styleClass = '', ...rest } = props;

  return (
    <ChakraRgbaColorPicker
      sx={{
        '.react-colorful__hue-pointer': colorPickerStyles,
        '.react-colorful__saturation-pointer': colorPickerStyles,
        '.react-colorful__alpha-pointer': colorPickerStyles,
      }}
      className={styleClass}
      {...rest}
    />
  );
};

export default memo(IAIColorPicker);
