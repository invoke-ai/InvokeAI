import { RgbaColorPicker } from 'react-colorful';
import { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types';

type IAIColorPickerProps = ColorPickerBaseProps<RgbaColor> & {
  styleClass?: string;
};

const IAIColorPicker = (props: IAIColorPickerProps) => {
  const { styleClass, ...rest } = props;

  return (
    <RgbaColorPicker
      className={`invokeai__color-picker ${styleClass}`}
      {...rest}
    />
  );
};

export default IAIColorPicker;
