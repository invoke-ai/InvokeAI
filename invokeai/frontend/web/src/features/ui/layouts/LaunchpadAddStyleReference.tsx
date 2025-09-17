import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToCroppableImage } from 'features/controlLayers/store/util';
import { addGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { LaunchpadButton } from 'features/ui/layouts/LaunchpadButton';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadBold, PiUserCircleGearBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const dndTargetData = addGlobalReferenceImageDndTarget.getData();

export const LaunchpadAddStyleReference = memo((props: { extraAction?: () => void }) => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();

  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          const config = getDefaultRefImageConfig(getState);
          config.image = imageDTOToCroppableImage(imageDTO);
          dispatch(refImageAdded({ overrides: { config } }));
          props.extraAction?.();
        },
        allowMultiple: false,
      }) as const,
    [dispatch, getState, props]
  );

  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8}>
      <Icon as={PiUserCircleGearBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">{t('ui.launchpad.addStyleRef.title')}</Heading>
        <Text>{t('ui.launchpad.addStyleRef.description')}</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiUploadBold />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <DndDropTarget dndTarget={addGlobalReferenceImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </LaunchpadButton>
  );
});
LaunchpadAddStyleReference.displayName = 'LaunchpadAddStyleReference';
