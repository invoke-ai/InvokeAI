import { Flex, Text } from '@chakra-ui/react';
import { memo, useMemo } from 'react';

export const ratioToCSSString = (
  ratio: AspectRatio,
  orientation: Orientation
) => {
  if (orientation === 'portrait') {
    return `${ratio[0]}/${ratio[1]}`;
  }
  return `${ratio[1]}/${ratio[0]}`;
};

export const ratioToDisplayString = (
  ratio: AspectRatio,
  orientation: Orientation
) => {
  if (orientation === 'portrait') {
    return `${ratio[0]}:${ratio[1]}`;
  }
  return `${ratio[1]}:${ratio[0]}`;
};

type AspectRatioPreviewProps = {
  ratio: AspectRatio;
  orientation: Orientation;
  size: string;
};

export type AspectRatio = [number, number];

export type Orientation = 'portrait' | 'landscape';

const AspectRatioPreview = (props: AspectRatioPreviewProps) => {
  const { ratio, size, orientation } = props;

  const ratioCSSString = useMemo(() => {
    if (orientation === 'portrait') {
      return `${ratio[0]}/${ratio[1]}`;
    }
    return `${ratio[1]}/${ratio[0]}`;
  }, [ratio, orientation]);

  const ratioDisplayString = useMemo(() => `${ratio[0]}:${ratio[1]}`, [ratio]);

  return (
    <Flex
      sx={{
        alignItems: 'center',
        justifyContent: 'center',
        w: size,
        h: size,
      }}
    >
      <Flex
        sx={{
          alignItems: 'center',
          justifyContent: 'center',
          bg: 'base.700',
          color: 'base.400',
          borderRadius: 'base',
          aspectRatio: ratioCSSString,
          objectFit: 'contain',
          ...(orientation === 'landscape' ? { h: 'full' } : { w: 'full' }),
        }}
      >
        <Text sx={{ size: 'xs', userSelect: 'none' }}>
          {ratioDisplayString}
        </Text>
      </Flex>
    </Flex>
  );
};

export default memo(AspectRatioPreview);
