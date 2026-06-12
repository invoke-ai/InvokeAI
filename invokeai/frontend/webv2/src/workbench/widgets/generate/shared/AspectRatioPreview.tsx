import { Box, Flex } from '@chakra-ui/react';

type Props = {
  ratio: number; // e.g. 16 / 9
  boxSize?: string | number;
};

export const AspectRatioPreview = ({ boxSize = '8', ratio }: Props) => {
  const normalizedRatio = Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
  const isWide = normalizedRatio >= 1;

  return (
    <Flex as="span" alignItems="center" boxSize={boxSize} flexShrink={0} justifyContent="center" p="0.5">
      <Box
        as="span"
        aria-hidden="true"
        bg="bg.subtle"
        borderColor="fg.muted"
        borderStyle="dashed"
        borderWidth="1px"
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
