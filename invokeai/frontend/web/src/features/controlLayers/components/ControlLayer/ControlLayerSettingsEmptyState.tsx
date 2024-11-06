import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useMemo } from 'react';
import { Trans } from 'react-i18next';
import type { PostUploadAction } from 'services/api/types';

export const ControlLayerSettingsEmptyState = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const postUploadAction = useMemo<PostUploadAction>(
    () => ({ type: 'REPLACE_LAYER_WITH_IMAGE', entityIdentifier }),
    [entityIdentifier]
  );
  const uploadApi = useImageUploadButton({ postUploadAction });
  const onClickGalleryButton = useCallback(() => {
    dispatch(activeTabCanvasRightPanelChanged('gallery'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full" p={4}>
      <Text textAlign="center" color="base.300">
        <Trans
          i18nKey="controlLayers.controlLayerEmptyState"
          components={{
            UploadButton: (
              <Button
                isDisabled={isBusy}
                size="sm"
                variant="link"
                color="base.300"
                {...uploadApi.getUploadButtonProps()}
              />
            ),
            GalleryButton: (
              <Button onClick={onClickGalleryButton} isDisabled={isBusy} size="sm" variant="link" color="base.300" />
            ),
          }}
        />
      </Text>
      <input {...uploadApi.getUploadInputProps()} />
    </Flex>
  );
});

ControlLayerSettingsEmptyState.displayName = 'ControlLayerSettingsEmptyState';
