import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { SetRegionalGuidanceReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { setRegionalGuidanceReferenceImage } from 'features/imageActions/actions';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

type Props = {
  referenceImageId: string;
};

export const RegionalGuidanceIPAdapterSettingsEmptyState = memo(({ referenceImageId }: Props) => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      setRegionalGuidanceReferenceImage({ imageDTO, entityIdentifier, referenceImageId, dispatch });
    },
    [dispatch, entityIdentifier, referenceImageId]
  );
  const uploadApi = useImageUploadButton({ onUpload, allowMultiple: false });
  const onClickGalleryButton = useCallback(() => {
    dispatch(activeTabCanvasRightPanelChanged('gallery'));
  }, [dispatch]);

  const dndTargetData = useMemo<SetRegionalGuidanceReferenceImageDndTargetData>(
    () =>
      setRegionalGuidanceReferenceImageDndTarget.getData({
        entityIdentifier,
        referenceImageId,
      }),
    [entityIdentifier, referenceImageId]
  );

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full" p={4}>
      <Text textAlign="center" color="base.300">
        <Trans
          i18nKey="controlLayers.referenceImageEmptyState"
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
      <DndDropTarget
        dndTarget={setRegionalGuidanceReferenceImageDndTarget}
        dndTargetData={dndTargetData}
        label={t('controlLayers.useImage')}
        isDisabled={isBusy}
      />
    </Flex>
  );
});

RegionalGuidanceIPAdapterSettingsEmptyState.displayName = 'RegionalGuidanceIPAdapterSettingsEmptyState';
