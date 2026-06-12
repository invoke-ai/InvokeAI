import { Box, Flex, Text } from '@chakra-ui/react';
import type { ReactNode } from 'react';

import type { CanvasPlacementContract, GeneratedImageContract } from '../../types';

const toPercent = (value: number, max: number) => `${(value / max) * 100}%`;

export const CanvasDocumentFrame = ({
  children,
  documentHeight,
  documentWidth,
  hasContent,
}: {
  children: ReactNode;
  documentHeight: number;
  documentWidth: number;
  hasContent: boolean;
}) => (
  <Box
    bg="bg.emphasized"
    borderWidth="1px"
    borderColor="border.emphasized"
    boxShadow="lg"
    maxH="calc(100vh - 15rem)"
    maxW="calc(100% - 4rem)"
    overflow="hidden"
    position="relative"
    rounded="md"
    w="min(78vw, 960px)"
    {...(hasContent ? { aspectRatio: `${documentWidth} / ${documentHeight}` } : {})}
  >
    {children}
  </Box>
);

export const CanvasPlaneImage = ({
  image,
  isStaged,
  opacity,
  placement,
  planeHeight,
  planeWidth,
}: {
  image: GeneratedImageContract;
  isStaged?: boolean;
  opacity: number;
  placement: CanvasPlacementContract;
  planeHeight: number;
  planeWidth: number;
}) => (
  <Box
    borderWidth={isStaged ? '2px' : '0'}
    borderColor="accent.solid"
    boxShadow={isStaged ? '0 0 0 1px {colors.accent.solid}, 0 18px 60px rgba(0,0,0,0.38)' : undefined}
    left={toPercent(placement.x, planeWidth)}
    opacity={opacity}
    overflow="hidden"
    position="absolute"
    top={toPercent(placement.y, planeHeight)}
    w={toPercent(placement.width, planeWidth)}
    h={toPercent(placement.height, planeHeight)}
    zIndex={isStaged ? 2 : 1}
  >
    <img
      alt={isStaged ? `Staging preview ${image.imageName}` : image.imageName}
      src={image.imageUrl}
      style={{ display: 'block', height: '100%', objectFit: 'contain', width: '100%' }}
    />
  </Box>
);

export const EmptyCanvasFrame = () => (
  <Flex
    align="center"
    bg="bg.emphasized"
    borderWidth="1px"
    borderColor="border.emphasized"
    color="fg.subtle"
    h="min(56vh, 34rem)"
    justify="center"
    rounded="md"
  >
    <Text fontSize="sm">Canvas layer stack is empty.</Text>
  </Flex>
);

export const ToolScrubber = () => (
  <Box
    bg="bg"
    borderWidth="1px"
    borderColor="border.emphasized"
    h="22rem"
    left="2.5"
    position="absolute"
    rounded="md"
    top="2.5"
    w="8"
    zIndex="1"
  />
);
