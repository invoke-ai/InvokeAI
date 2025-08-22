import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import { ParamResolution } from './ParamResolution';
import { VideoDimensionsAspectRatioSelect } from './VideoDimensionsAspectRatioSelect';
import { VideoDimensionsPreview } from './VideoDimensionsPreview';

export const VideoDimensions = memo(() => {
  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <ParamResolution />
        <VideoDimensionsAspectRatioSelect />
      </Flex>
      <Flex w="108px" h="108px" flexShrink={0} flexGrow={0} alignItems="center" justifyContent="center" py={4}>
        <VideoDimensionsPreview />
      </Flex>
    </Flex>
  );
});

VideoDimensions.displayName = 'VideoDimensions';
