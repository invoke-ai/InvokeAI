import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Flex, Icon, Image, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { GalleryProgressItem } from 'features/gallery/store/galleryProgressStore';
import { $galleryProgressItems } from 'features/gallery/store/galleryProgressStore';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { memo, useMemo } from 'react';
import { PiClockBold } from 'react-icons/pi';

const selectShouldAntialiasProgressImage = createSelector(
  selectSystemSlice,
  (system) => system.shouldAntialiasProgressImage
);

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
};

interface Props {
  itemId: number;
}

export const GalleryProgressTile = memo(({ itemId }: Props) => {
  const allItems = useStore($galleryProgressItems);
  const item = allItems[itemId];

  if (!item) {
    return null;
  }

  if (item.status === 'in_progress' && item.progressImage) {
    return <InProgressTile item={item} />;
  }

  return <PendingTile item={item} />;
});

GalleryProgressTile.displayName = 'GalleryProgressTile';

const PendingTile = memo((_props: { item: GalleryProgressItem }) => {
  return (
    <Flex
      w="full"
      h="full"
      alignItems="center"
      justifyContent="center"
      bg="base.850"
      flexDir="column"
      gap={2}
      borderWidth={1}
      borderColor="base.700"
      borderRadius="base"
      aspectRatio="1/1"
    >
      <Icon as={PiClockBold} boxSize={8} color="base.500" />
      <Text fontSize="xs" color="base.500" fontWeight="semibold">
        Queued
      </Text>
    </Flex>
  );
});

PendingTile.displayName = 'PendingTile';

const InProgressTile = memo(({ item }: { item: GalleryProgressItem }) => {
  const shouldAntialiasProgressImage = useAppSelector(selectShouldAntialiasProgressImage);

  const imageSx = useMemo<SystemStyleObject>(
    () => ({
      imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
    }),
    [shouldAntialiasProgressImage]
  );

  return (
    <Flex position="relative" w="full" h="full" aspectRatio="1/1" borderRadius="base" overflow="hidden">
      {item.progressImage && (
        <Image
          src={item.progressImage.dataURL}
          w="full"
          h="full"
          objectFit="cover"
          borderRadius="base"
          draggable={false}
          sx={imageSx}
        />
      )}
      <Flex position="absolute" top={1} insetInlineEnd={1} bg="blackAlpha.700" borderRadius="full" p={0.5}>
        <CircularProgress
          value={item.percentage !== null ? item.percentage * 100 : undefined}
          isIndeterminate={item.percentage === null}
          size="20px"
          thickness={14}
          trackColor="transparent"
          color="invokeBlue.500"
          sx={circleStyles}
        />
      </Flex>
      <Flex
        position="absolute"
        bottom={0}
        insetInlineStart={0}
        insetInlineEnd={0}
        bg="blackAlpha.700"
        px={1}
        py={0.5}
        borderBottomRadius="base"
      >
        <Text fontSize="2xs" color="base.300" noOfLines={1}>
          {item.message ?? 'Generating...'}
        </Text>
      </Flex>
    </Flex>
  );
});

InProgressTile.displayName = 'InProgressTile';
