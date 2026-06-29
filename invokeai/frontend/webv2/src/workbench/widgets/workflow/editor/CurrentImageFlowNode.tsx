import type { GeneratedImageContract } from '@workbench/types';
import type { NodeProps } from '@xyflow/react';

import { Box, Flex, Text } from '@chakra-ui/react';
import { useProgressImage } from '@workbench/backend/progressImageStore';
import { StreamingImageFrame } from '@workbench/images/StreamingImageFrame';
import { imageUrlToStreamingSource, progressImageToStreamingSource } from '@workbench/images/streamingImageSource';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { memo } from 'react';

import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

import { getWorkflowNodeChromeProps } from './nodeChrome';

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
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const progressImage = useProgressImage();
  const node = data.documentNode;
  const latestImage = getLatestImage(galleryValues);

  return (
    <Box bg="bg" fontSize="xs" overflow="hidden" rounded="lg" w="20rem" {...getWorkflowNodeChromeProps({ selected })}>
      <Flex align="center" bg="bg.subtle" borderBottomWidth="1px" borderColor="border.subtle" px="3" py="1.5">
        <Text fontWeight="700">{node.data.label || 'Current Image'}</Text>
        {progressImage ? (
          <Text color="brand.solid" fontSize="2xs" ms="auto">
            generating…
          </Text>
        ) : null}
      </Flex>
      <StreamingImageFrame
        fallbackImage={imageUrlToStreamingSource({
          alt: latestImage?.imageName ?? 'Current image',
          height: latestImage?.height,
          kind: 'fallback',
          src: latestImage?.imageUrl,
          width: latestImage?.width,
        })}
        fit="contain"
        h="18rem"
        liveImage={progressImageToStreamingSource(progressImage)}
        w="full"
      >
        <Flex align="center" color="fg.subtle" fontSize="2xs" h="full" justify="center" px="4" textAlign="center">
          No image yet — the latest generation will appear here.
        </Flex>
      </StreamingImageFrame>
    </Box>
  );
};

export const CurrentImageFlowNode = memo(CurrentImageFlowNodeComponent);
