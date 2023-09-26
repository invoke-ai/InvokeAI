import { Box, ChakraProps } from '@chakra-ui/react';
import { memo } from 'react';
import { RgbaColorPicker } from 'react-colorful';
import { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types';

type IAIColorPickerProps = ColorPickerBaseProps<RgbaColor>;

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
  return (
    <Box sx={sx}>
      <RgbaColorPicker {...props} />
    </Box>
  );
};

export default memo(IAIColorPicker);
