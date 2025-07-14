import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { usePullBboxIntoLayer } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { replaceCanvasEntityObjectsWithImage } from 'features/imageActions/actions';
import { memo, useCallback, useMemo } from 'react';
import { Trans } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

export const ControlLayerSettingsEmptyState = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const { dispatch, getState } = useAppStore();
  const isBusy = useCanvasIsBusy();
  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      replaceCanvasEntityObjectsWithImage({ imageDTO, entityIdentifier, dispatch, getState });
    },
    [dispatch, entityIdentifier, getState]
  );
  const uploadApi = useImageUploadButton({ onUpload, allowMultiple: false });
  const pullBboxIntoLayer = usePullBboxIntoLayer(entityIdentifier);

  const components = useMemo(
    () => ({
      UploadButton: (
        <Button isDisabled={isBusy} size="sm" variant="link" color="base.300" {...uploadApi.getUploadButtonProps()} />
      ),
      PullBboxButton: (
        <Button onClick={pullBboxIntoLayer} isDisabled={isBusy} size="sm" variant="link" color="base.300" />
      ),
    }),
    [isBusy, pullBboxIntoLayer, uploadApi]
  );

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full" p={4}>
      <Text textAlign="center" color="base.300">
        <Trans i18nKey="controlLayers.controlLayerEmptyState" components={components} />
      </Text>
      <input {...uploadApi.getUploadInputProps()} />
    </Flex>
  );
});

ControlLayerSettingsEmptyState.displayName = 'ControlLayerSettingsEmptyState';
