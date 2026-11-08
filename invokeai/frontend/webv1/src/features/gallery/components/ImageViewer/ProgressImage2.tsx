import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { memo, useMemo } from 'react';

const selectShouldAntialiasProgressImage = createSelector(
  selectSystemSlice,
  (system) => system.shouldAntialiasProgressImage
);

export const ProgressImage = memo(({ progressImage }: { progressImage: ProgressImageType }) => {
  const shouldAntialiasProgressImage = useAppSelector(selectShouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  return (
    <Flex width="full" height="full" alignItems="center" justifyContent="center" minW={0} minH={0}>
      <Image
        src={progressImage.dataURL}
        width={progressImage.width}
        height={progressImage.height}
        draggable={false}
        data-testid="progress-image"
        objectFit="contain"
        maxWidth="full"
        maxHeight="full"
        borderRadius="base"
        sx={sx}
        minH={0}
        minW={0}
      />
    </Flex>
  );
});

ProgressImage.displayName = 'ProgressImage';
