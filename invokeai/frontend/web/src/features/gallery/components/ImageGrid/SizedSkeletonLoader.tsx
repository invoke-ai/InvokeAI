import { Skeleton } from '@invoke-ai/ui-library';
import { memo } from 'react';

type Props = {
  width: number;
  height: number;
};

export const SizedSkeletonLoader = memo(({ width, height }: Props) => {
  return <Skeleton w={`${width}px`} h="auto" objectFit="contain" aspectRatio={`${width}/${height}`} />;
});

SizedSkeletonLoader.displayName = 'SizedSkeletonLoader';
