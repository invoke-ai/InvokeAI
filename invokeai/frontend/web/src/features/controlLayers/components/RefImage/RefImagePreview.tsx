import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, IconButton, Image, Skeleton, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { round } from 'es-toolkit/compat';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import {
  refImageSelected,
  selectIsRefImagePanelOpen,
  selectSelectedRefEntityId,
} from 'features/controlLayers/store/refImagesSlice';
import { isIPAdapterConfig } from 'features/controlLayers/store/types';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { PiExclamationMarkBold, PiEyeSlashBold, PiImageBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const baseSx: SystemStyleObject = {
  '&[data-is-open="true"]': {
    borderColor: 'invokeBlue.300',
  },
  '&[data-is-disabled="true"]': {
    opacity: 0.4,
  },
};

const weightDisplaySx: SystemStyleObject = {
  pointerEvents: 'none',
  transitionProperty: 'opacity',
  transitionDuration: 'normal',
  opacity: 0,
  '&[data-visible="true"]': {
    opacity: 1,
  },
};

const getImageSxWithWeight = (weight: number): SystemStyleObject => {
  const fillPercentage = Math.max(0, Math.min(100, weight * 100));

  return {
    ...baseSx,
    '&[data-is-disabled="true"]': {
      opacity: 0.4,
    },
    _after: {
      content: '""',
      position: 'absolute',
      inset: 0,
      background: `linear-gradient(to top, transparent ${fillPercentage}%, rgba(0, 0, 0, 0.8) ${fillPercentage}%)`,
      pointerEvents: 'none',
      borderRadius: 'base',
    },
  };
};

export const RefImagePreview = memo(() => {
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const entity = useRefImageEntity(id);
  const selectedEntityId = useAppSelector(selectSelectedRefEntityId);
  const isPanelOpen = useAppSelector(selectIsRefImagePanelOpen);
  const [showWeightDisplay, setShowWeightDisplay] = useState(false);
  const { data: imageDTO } = useGetImageDTOQuery(entity.config.image?.image_name ?? skipToken);

  const sx = useMemo(() => {
    if (!isIPAdapterConfig(entity.config)) {
      return baseSx;
    }
    return getImageSxWithWeight(entity.config.weight);
  }, [entity.config]);

  useEffect(() => {
    if (!isIPAdapterConfig(entity.config)) {
      return;
    }
    setShowWeightDisplay(true);
    const timeout = window.setTimeout(() => {
      setShowWeightDisplay(false);
    }, 1000);
    return () => {
      window.clearTimeout(timeout);
    };
  }, [entity.config]);

  const onClick = useCallback(() => {
    dispatch(refImageSelected({ id }));
  }, [dispatch, id]);

  if (!entity.config.image) {
    return (
      <IconButton
        aria-label="Select Ref Image"
        h="full"
        variant="ghost"
        aspectRatio="1/1"
        borderWidth={1}
        borderStyle="solid"
        borderColor="error.300"
        borderRadius="base"
        icon={<PiImageBold />}
        colorScheme="error"
        onClick={onClick}
        flexShrink={0}
        data-is-open={selectedEntityId === id && isPanelOpen}
        data-is-error={true}
        data-is-disabled={!entity.isEnabled}
        sx={sx}
      />
    );
  }
  return (
    <Flex
      position="relative"
      borderWidth={1}
      borderStyle="solid"
      borderRadius="base"
      aspectRatio="1/1"
      maxW="full"
      maxH="full"
      flexShrink={0}
      sx={sx}
      data-is-open={selectedEntityId === id && isPanelOpen}
      data-is-error={!entity.config.model}
      data-is-disabled={!entity.isEnabled}
      role="button"
      onClick={onClick}
      cursor="pointer"
    >
      <Image
        src={imageDTO?.thumbnail_url}
        objectFit="contain"
        aspectRatio="1/1"
        height={imageDTO?.height}
        fallback={<Skeleton h="full" aspectRatio="1/1" />}
        maxW="full"
        maxH="full"
        borderRadius="base"
      />
      {isIPAdapterConfig(entity.config) && (
        <Flex
          position="absolute"
          inset={0}
          fontWeight="semibold"
          alignItems="center"
          justifyContent="center"
          zIndex={1}
          data-visible={showWeightDisplay}
          sx={weightDisplaySx}
        >
          <Text filter="drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 2px rgba(0, 0, 0, 1))">
            {`${round(entity.config.weight * 100, 2)}%`}
          </Text>
        </Flex>
      )}
      {!entity.config.model && (
        <Icon
          position="absolute"
          top="50%"
          left="50%"
          transform="translateX(-50%) translateY(-50%)"
          filter="drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 2px rgba(0, 0, 0, 1))"
          color="error.500"
          boxSize={6}
          as={PiExclamationMarkBold}
        />
      )}
      {!entity.isEnabled && (
        <Icon
          position="absolute"
          top="50%"
          left="50%"
          transform="translateX(-50%) translateY(-50%)"
          filter="drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 2px rgba(0, 0, 0, 1))"
          color="base.100"
          boxSize={6}
          as={PiEyeSlashBold}
        />
      )}
    </Flex>
  );
});
RefImagePreview.displayName = 'RefImagePreview';
