import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { usePullBboxIntoGlobalReferenceImage } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { SetGlobalReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { setGlobalReferenceImage } from 'features/imageActions/actions';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

export const RefImageNoImageState = memo(() => {
  const { t } = useTranslation();
  const id = useRefImageIdContext();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      setGlobalReferenceImage({ imageDTO, id, dispatch });
    },
    [dispatch, id]
  );
  const uploadApi = useImageUploadButton({ onUpload, allowMultiple: false });
  const onClickGalleryButton = useCallback(() => {
    dispatch(activeTabCanvasRightPanelChanged('gallery'));
  }, [dispatch]);
  const pullBboxIntoIPAdapter = usePullBboxIntoGlobalReferenceImage(id);

  const dndTargetData = useMemo<SetGlobalReferenceImageDndTargetData>(
    () => setGlobalReferenceImageDndTarget.getData({ id }),
    [id]
  );

  const components = useMemo(
    () => ({
      UploadButton: (
        <Button isDisabled={isBusy} size="sm" variant="link" color="base.300" {...uploadApi.getUploadButtonProps()} />
      ),
      GalleryButton: (
        <Button onClick={onClickGalleryButton} isDisabled={isBusy} size="sm" variant="link" color="base.300" />
      ),
      PullBboxButton: (
        <Button onClick={pullBboxIntoIPAdapter} isDisabled={isBusy} size="sm" variant="link" color="base.300" />
      ),
    }),
    [isBusy, onClickGalleryButton, pullBboxIntoIPAdapter, uploadApi]
  );

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full" p={4}>
      <Text textAlign="center" color="base.300">
        <Trans i18nKey="controlLayers.referenceImageEmptyState" components={components} />
      </Text>
      <input {...uploadApi.getUploadInputProps()} />
      <DndDropTarget
        dndTarget={setGlobalReferenceImageDndTarget}
        dndTargetData={dndTargetData}
        label={t('controlLayers.useImage')}
        isDisabled={isBusy}
      />
    </Flex>
  );
});

RefImageNoImageState.displayName = 'RefImageNoImageState';
