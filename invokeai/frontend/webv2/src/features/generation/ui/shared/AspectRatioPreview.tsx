import { Box, Flex } from '@chakra-ui/react';

type Props = {
  /** Width over height, e.g. 16 / 9; null draws a dashed square standing in for a ratio not yet valid. */
  ratio: number | null;
  boxSize?: string | number;
};

/** A ratio's shape, fitted inside a square; decorative, since its ratio is always named beside it. */
export const AspectRatioPreview = ({ boxSize = '8', ratio }: Props) => {
  const isPlaceholder = ratio === null || !Number.isFinite(ratio) || ratio <= 0;
  const normalizedRatio = isPlaceholder ? 1 : ratio;
  const isWide = normalizedRatio >= 1;

  return (
    <Flex aria-hidden as="span" alignItems="center" boxSize={boxSize} flexShrink={0} justifyContent="center" p="0.5">
      <Box
        as="span"
        bg={isPlaceholder ? 'transparent' : 'bg.subtle'}
        borderColor={isPlaceholder ? 'border.emphasized' : 'fg.muted'}
        borderStyle={isPlaceholder ? 'dashed' : 'solid'}
        borderWidth="1px"
        data-placeholder={isPlaceholder || undefined}
        data-shape=""
        display="block"
        h={isWide ? `${100 / normalizedRatio}%` : 'full'}
        minH="1px"
        minW="1px"
        rounded="2px"
        w={isWide ? 'full' : `${normalizedRatio * 100}%`}
      />
    </Flex>
  );
};
