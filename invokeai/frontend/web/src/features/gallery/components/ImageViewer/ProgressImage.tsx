import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Image } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { memo, useMemo } from 'react';
import { PiPulseBold } from 'react-icons/pi';
import { useIsNonPromptExpansionGenerationInProgress } from 'services/api/endpoints/queue';
import { $lastProgressImage } from 'services/events/stores';

const selectShouldAntialiasProgressImage = createSelector(
  selectSystemSlice,
  (system) => system.shouldAntialiasProgressImage
);

export const ProgressImage = memo(() => {
  const isGenerationInProgress = useIsNonPromptExpansionGenerationInProgress();
  const progressImage = useStore($lastProgressImage);
  const shouldAntialiasProgressImage = useAppSelector(selectShouldAntialiasProgressImage);

  const sx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  if (!isGenerationInProgress) {
    return (
      <Flex width="full" height="full" alignItems="center" justifyContent="center">
        <IAINoContentFallback icon={PiPulseBold} label="No Generation in Progress" />
      </Flex>
    );
  }

  if (!progressImage) {
    return (
      <Flex width="full" height="full" position="relative" minW={0} minH={0}>
        <Badge
          position="absolute"
          top={2}
          left={2}
          color="base.300"
          borderColor="base.700"
          borderWidth={1}
          bg="base.900"
          opacity="0.8"
          fontSize="sm"
          fontWeight="semibold"
          zIndex="docked"
          pointerEvents="none"
          borderRadius="base"
        >
          Waiting for Image
        </Badge>
      </Flex>
    );
  }

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
