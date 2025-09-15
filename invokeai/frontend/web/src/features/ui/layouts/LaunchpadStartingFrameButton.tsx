import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { imageDTOToCroppableImage } from 'features/controlLayers/store/util';
import { videoFrameFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { startingFrameImageChanged } from 'features/parameters/store/videoSlice';
import { LaunchpadButton } from 'features/ui/layouts/LaunchpadButton';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadBold, PiVideoBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const dndTargetData = videoFrameFromImageDndTarget.getData({ frame: 'start' });

export const LaunchpadStartingFrameButton = memo((props: { extraAction?: () => void }) => {
  const { t } = useTranslation();
  const { dispatch } = useAppStore();

  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          dispatch(startingFrameImageChanged(imageDTOToCroppableImage(imageDTO)));
          props.extraAction?.();
        },
        allowMultiple: false,
      }) as const,
    [dispatch, props]
  );

  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8}>
      <Icon as={PiVideoBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">{t('ui.launchpad.addStartingFrame.title')}</Heading>
        <Text>{t('ui.launchpad.addStartingFrame.description')}</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiUploadBold />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <DndDropTarget dndTarget={videoFrameFromImageDndTarget} dndTargetData={dndTargetData} label="Drop" />
    </LaunchpadButton>
  );
});

LaunchpadStartingFrameButton.displayName = 'LaunchpadStartingFrameButton';
