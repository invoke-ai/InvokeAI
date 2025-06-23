import { Flex } from '@invoke-ai/ui-library';
import { ProgressImage } from 'features/gallery/components/ImageViewer/ProgressImage2';
import { ProgressIndicator } from 'features/gallery/components/ImageViewer/ProgressIndicator2';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { memo } from 'react';
import type { S } from 'services/api/types';

export const Progress = memo(
  ({
    progressEvent,
    progressImage,
  }: {
    progressEvent: S['InvocationProgressEvent'];
    progressImage: ProgressImageType;
  }) => (
    <Flex position="relative" flexDir="column" w="full" h="full" overflow="hidden" p={2}>
      <ProgressImage progressImage={progressImage} />
      <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
    </Flex>
  )
);
Progress.displayName = 'Progress';
