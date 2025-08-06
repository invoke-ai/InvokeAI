import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { newCanvasFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiRectangleDashedBold, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { LaunchpadButton } from './LaunchpadButton';

const NEW_CANVAS_OPTIONS = { type: 'control_layer', withResize: true } as const;

const dndTargetData = newCanvasFromImageDndTarget.getData(NEW_CANVAS_OPTIONS);

export const LaunchpadUseALayoutImageButton = memo((props: { extraAction?: () => void }) => {
  const { t } = useTranslation();
  const { getState, dispatch } = useAppStore();
  const isStaging = useCanvasIsStaging();

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      newCanvasFromImage({ imageDTO, getState, dispatch, ...NEW_CANVAS_OPTIONS });
      props.extraAction?.();
    },
    [dispatch, getState, props]
  );
  const uploadApi = useImageUploadButton({ allowMultiple: false, onUpload });

  return (
    <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8} isDisabled={isStaging}>
      <Icon as={PiRectangleDashedBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">{t('ui.launchpad.useALayoutImage.title')}</Heading>
        <Text>{t('ui.launchpad.useALayoutImage.description')}</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiUploadBold />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <DndDropTarget
        dndTarget={newCanvasFromImageDndTarget}
        dndTargetData={dndTargetData}
        label="Drop"
        isDisabled={isStaging}
      />
    </LaunchpadButton>
  );
});
LaunchpadUseALayoutImageButton.displayName = 'LaunchpadUseALayoutImageButton';
