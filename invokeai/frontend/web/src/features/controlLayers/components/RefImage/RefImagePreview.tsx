import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Icon, IconButton, Skeleton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { round } from 'es-toolkit/compat';
import { useRefImageDnd } from 'features/controlLayers/components/RefImage/useRefImageDnd';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import {
  refImageSelected,
  selectIsRefImagePanelOpen,
  selectSelectedRefEntityId,
} from 'features/controlLayers/store/refImagesSlice';
import { isIPAdapterConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiExclamationMarkBold, PiEyeSlashBold, PiImageBold } from 'react-icons/pi';
import { useImageDTOFromCroppableImage } from 'services/api/endpoints/images';
import { isExternalApiModelConfig } from 'services/api/types';

import { RefImageWarningTooltipContent } from './RefImageWarningTooltipContent';

const baseSx: SystemStyleObject = {
  '&[data-is-open="true"]': {
    borderColor: 'invokeBlue.300',
  },
  '&[data-is-disabled="true"]': {
    img: {
      opacity: 0.4,
      filter: 'grayscale(100%)',
    },
  },
  '&[data-is-error="true"]': {
    borderColor: 'error.500',
    img: {
      opacity: 0.4,
      filter: 'grayscale(100%)',
    },
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
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const entity = useRefImageEntity(id);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const selectedEntityId = useAppSelector(selectSelectedRefEntityId);
  const isPanelOpen = useAppSelector(selectIsRefImagePanelOpen);
  const [showWeightDisplay, setShowWeightDisplay] = useState(false);
  const isExternalModel = !!mainModelConfig && isExternalApiModelConfig(mainModelConfig);
  const dndRef = useRef<HTMLDivElement>(null);
  const [dndListState, isDragging] = useRefImageDnd(dndRef, id);

  const imageDTO = useImageDTOFromCroppableImage(entity.config.image);

  const sx = useMemo(() => {
    if (!isIPAdapterConfig(entity.config) || isExternalModel) {
      return baseSx;
    }
    return getImageSxWithWeight(entity.config.weight);
  }, [entity.config, isExternalModel]);

  useEffect(() => {
    if (!isIPAdapterConfig(entity.config) || isExternalModel) {
      return;
    }
    setShowWeightDisplay(true);
    const timeout = window.setTimeout(() => {
      setShowWeightDisplay(false);
    }, 1000);
    return () => {
      window.clearTimeout(timeout);
    };
  }, [entity.config, isExternalModel]);

  const warnings = useMemo(() => {
    return getGlobalReferenceImageWarnings(entity, mainModelConfig);
  }, [entity, mainModelConfig]);

  const onClick = useCallback(() => {
    dispatch(refImageSelected({ id }));
  }, [dispatch, id]);

  if (!entity.config.image) {
    return (
      <Box
        ref={dndRef}
        position="relative"
        h="full"
        flexShrink={0}
        opacity={isDragging ? 0.3 : 1}
        data-ref-image-id={id}
      >
        <IconButton
          aria-label={t('controlLayers.selectRefImage')}
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
        <DndListDropIndicator dndState={dndListState} />
      </Box>
    );
  }
  return (
    <Box ref={dndRef} position="relative" h="full" flexShrink={0} opacity={isDragging ? 0.3 : 1} data-ref-image-id={id}>
      <Tooltip label={warnings.length > 0 ? <RefImageWarningTooltipContent warnings={warnings} /> : undefined}>
        <Flex
          position="relative"
          borderWidth={1}
          borderStyle="solid"
          borderRadius="base"
          aspectRatio="1/1"
          maxW="full"
          h="full"
          maxH="full"
          flexShrink={0}
          sx={sx}
          data-is-open={selectedEntityId === id && isPanelOpen}
          data-is-error={warnings.length > 0}
          data-is-disabled={!entity.isEnabled}
          role="button"
          onClick={onClick}
          cursor="pointer"
          overflow="hidden"
        >
          {imageDTO ? (
            <img
              src={imageDTO.image_url}
              style={{ objectFit: 'contain', aspectRatio: '1 / 1', maxWidth: '100%', maxHeight: '100%' }}
              height={imageDTO.height}
              alt={imageDTO.image_name}
              draggable={false}
            />
          ) : (
            <Skeleton h="full" aspectRatio="1/1" />
          )}
          {isIPAdapterConfig(entity.config) && !isExternalModel && (
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
          {!entity.isEnabled && (
            <Icon
              position="absolute"
              top="50%"
              left="50%"
              transform="translateX(-50%) translateY(-50%)"
              filter="drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 2px rgba(0, 0, 0, 1))"
              color="base.300"
              boxSize={8}
              as={PiEyeSlashBold}
            />
          )}
          {entity.isEnabled && warnings.length > 0 && (
            <Icon
              position="absolute"
              top="50%"
              left="50%"
              transform="translateX(-50%) translateY(-50%)"
              filter="drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 2px rgba(0, 0, 0, 1))"
              color="error.500"
              boxSize={12}
              as={PiExclamationMarkBold}
            />
          )}
        </Flex>
      </Tooltip>
      <DndListDropIndicator dndState={dndListState} />
    </Box>
  );
});
RefImagePreview.displayName = 'RefImagePreview';
