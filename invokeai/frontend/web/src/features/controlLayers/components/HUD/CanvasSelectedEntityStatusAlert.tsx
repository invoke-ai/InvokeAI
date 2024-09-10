import { Box, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import type { Property } from 'csstype';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import {
  selectCanvasSlice,
  selectEntityOrThrow,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { atom } from 'nanostores';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningCircleFill } from 'react-icons/pi';

type ContentProps = {
  entityIdentifier: CanvasEntityIdentifier;
  adapter: CanvasEntityAdapter;
};

const $isFilteringFallback = atom(false);

type EntityStatus = {
  value: string;
  color?: Property.Color;
};

const CanvasSelectedEntityStatusAlertContent = memo(({ entityIdentifier, adapter }: ContentProps) => {
  const { t } = useTranslation();
  const title = useEntityTitle(entityIdentifier);
  const selectIsEnabled = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).isEnabled),
    [entityIdentifier]
  );
  const selectIsLocked = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntityOrThrow(canvas, entityIdentifier).isLocked),
    [entityIdentifier]
  );
  const isEnabled = useAppSelector(selectIsEnabled);
  const isLocked = useAppSelector(selectIsLocked);
  const isHidden = useEntityTypeIsHidden(entityIdentifier.type);
  const isFiltering = useStore(adapter.filterer?.$isFiltering ?? $isFilteringFallback);
  const isTransforming = useStore(adapter.transformer.$isTransforming);

  const status = useMemo<EntityStatus | null>(() => {
    if (isFiltering) {
      return {
        value: t('controlLayers.HUD.entityStatus.isFiltering'),
        color: 'invokeBlue.300',
      };
    }

    if (isTransforming) {
      return {
        value: t('controlLayers.HUD.entityStatus.isTransforming'),
        color: 'invokeBlue.300',
      };
    }

    if (isHidden) {
      return {
        value: t('controlLayers.HUD.entityStatus.isHidden'),
        color: 'invokePurple.300',
      };
    }

    if (isLocked) {
      return {
        value: t('controlLayers.HUD.entityStatus.isLocked'),
        color: 'invokeRed.300',
      };
    }

    if (!isEnabled) {
      return {
        value: t('controlLayers.HUD.entityStatus.isDisabled'),
        color: 'invokeRed.300',
      };
    }

    return null;
  }, [isFiltering, isTransforming, isHidden, isLocked, isEnabled, t]);

  if (!status) {
    return null;
  }

  return (
    <Box position="relative" shadow="dark-lg">
      <Flex
        position="absolute"
        top={0}
        right={0}
        left={0}
        bottom={0}
        bg={status.color}
        opacity={0.3}
        borderRadius="base"
        borderColor="whiteAlpha.400"
        borderWidth={1}
      />
      <Flex px={6} py={4} gap={6} alignItems="center" justifyContent="center">
        <Icon as={PiWarningCircleFill} />
        <Text as="span" h={8}>
          <Text as="span" fontWeight="semibold">
            {title}
          </Text>{' '}
          {status.value}
        </Text>
      </Flex>
    </Box>
  );
});

CanvasSelectedEntityStatusAlertContent.displayName = 'CanvasSelectedEntityStatusAlertContent';

export const CanvasSelectedEntityStatusAlert = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);

  if (!selectedEntityIdentifier || !adapter) {
    return null;
  }

  return <CanvasSelectedEntityStatusAlertContent entityIdentifier={selectedEntityIdentifier} adapter={adapter} />;
});

CanvasSelectedEntityStatusAlert.displayName = 'CanvasSelectedEntityStatusAlert';
