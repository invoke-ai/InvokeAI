import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex, forwardRef } from '@invoke-ai/ui-library';
import { useMemo } from 'react';
import type { RgbaColor, RgbColor } from 'react-colorful';

type Props = FlexProps & {
  previewColor: RgbColor | RgbaColor;
};

export const ColorPreview = forwardRef((props: Props, ref) => {
  const { previewColor, ...rest } = props;
  const colorString = useMemo(() => {
    if ('a' in previewColor) {
      return `rgba(${previewColor.r}, ${previewColor.g}, ${previewColor.b}, ${previewColor.a ?? 1})`;
    }
    return `rgba(${previewColor.r}, ${previewColor.g}, ${previewColor.b}, 1)`;
  }, [previewColor]);
  return <Flex ref={ref} w="full" h="full" borderRadius="base" backgroundColor={colorString} {...rest} />;
});
