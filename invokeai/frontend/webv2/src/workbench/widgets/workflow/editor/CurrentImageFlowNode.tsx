import { Box, Flex, Image, Text } from '@chakra-ui/react';
import type { NodeProps } from '@xyflow/react';
import { memo } from 'react';

import { useProgressImage } from '@workbench/backend/progressImageStore';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import type { GeneratedImageContract } from '@workbench/types';
import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

/**
 * The legacy `current_image` UI node: a live monitor inside the graph. Shows
 * the in-flight denoising preview while a run executes, and the most recently
 * generated image otherwise.
 */

const getLatestImage = (values: Record<string, unknown>): GeneratedImageContract | null => {
  const recentImages = Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];

  return recentImages[0] ?? null;
};

const CurrentImageFlowNodeComponent = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const galleryValues = useActiveProjectSelector((project) => project.widgetStates.gallery.values);
  const progressImage = useProgressImage();
  const node = data.documentNode;
  const latestImage = getLatestImage(galleryValues);
  const src = progressImage?.dataUrl ?? latestImage?.imageUrl ?? null;

  return (
    <Box
      bg="bg"
      borderColor={selected ? 'accent.solid' : 'border.emphasized'}
      borderWidth="1px"
      fontSize="xs"
      overflow="hidden"
      rounded="lg"
      shadow={selected ? 'md' : 'sm'}
      w="20rem"
    >
      <Flex align="center" bg="bg.subtle" borderBottomWidth="1px" borderColor="border.subtle" px="3" py="1.5">
        <Text fontWeight="700">{node.data.label || 'Current Image'}</Text>
        {progressImage ? (
          <Text color="brand.solid" fontSize="2xs" ms="auto">
            generating…
          </Text>
        ) : null}
      </Flex>
      {src ? (
        <Image alt="Current image" draggable={false} h="18rem" objectFit="contain" src={src} w="full" />
      ) : (
        <Flex align="center" color="fg.subtle" fontSize="2xs" h="18rem" justify="center" px="4" textAlign="center">
          No image yet — the latest generation will appear here.
        </Flex>
      )}
    </Box>
  );
};

export const CurrentImageFlowNode = memo(CurrentImageFlowNodeComponent);
