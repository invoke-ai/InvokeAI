import { Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { PostProcessingPopover } from 'features/parameters/components/PostProcessing/PostProcessingPopover';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiDotsThreeOutlineFill,
  PiFlowArrowBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

const CurrentImageButtons = () => {
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  if (!imageDTO) {
    return null;
  }

  return <CurrentImageButtonsContent imageDTO={imageDTO} />;
};

export default memo(CurrentImageButtons);

const CurrentImageButtonsContent = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const { t } = useTranslation();
  const hasTemplates = useStore($hasTemplates);
  const imageActions = useImageActions(imageDTO);
  const isStaging = useAppSelector(selectIsStaging);
  const isUpscalingEnabled = useFeatureStatus('upscaling');

  return (
    <>
      <Menu isLazy>
        <MenuButton
          as={IconButton}
          aria-label={t('parameters.imageActions')}
          tooltip={t('parameters.imageActions')}
          isDisabled={!imageDTO}
          variant="link"
          alignSelf="stretch"
          icon={<PiDotsThreeOutlineFill />}
        />
        <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
      </Menu>

      <Divider orientation="vertical" h={8} mx={2} />

      <IconButton
        icon={<PiFlowArrowBold />}
        tooltip={`${t('nodes.loadWorkflow')} (W)`}
        aria-label={`${t('nodes.loadWorkflow')} (W)`}
        isDisabled={!imageActions.hasWorkflow || !hasTemplates}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.loadWorkflow}
      />
      <IconButton
        icon={<PiArrowsCounterClockwiseBold />}
        tooltip={`${t('parameters.remixImage')} (R)`}
        aria-label={`${t('parameters.remixImage')} (R)`}
        isDisabled={!imageActions.hasMetadata}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.remix}
      />
      <IconButton
        icon={<PiQuotesBold />}
        tooltip={`${t('parameters.usePrompt')} (P)`}
        aria-label={`${t('parameters.usePrompt')} (P)`}
        isDisabled={!imageActions.hasPrompts}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallPrompts}
      />
      <IconButton
        icon={<PiPlantBold />}
        tooltip={`${t('parameters.useSeed')} (S)`}
        aria-label={`${t('parameters.useSeed')} (S)`}
        isDisabled={!imageActions.hasSeed}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallSeed}
      />
      <IconButton
        icon={<PiRulerBold />}
        tooltip={`${t('parameters.useSize')} (D)`}
        aria-label={`${t('parameters.useSize')} (D)`}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallSize}
        isDisabled={isStaging}
      />
      <IconButton
        icon={<PiAsteriskBold />}
        tooltip={`${t('parameters.useAll')} (A)`}
        aria-label={`${t('parameters.useAll')} (A)`}
        isDisabled={!imageActions.hasMetadata}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallAll}
      />

      {isUpscalingEnabled && <PostProcessingPopover imageDTO={imageDTO} />}

      <Divider orientation="vertical" h={8} mx={2} />

      <DeleteImageButton onClick={imageActions.delete} />
    </>
  );
});

CurrentImageButtonsContent.displayName = 'CurrentImageButtonsContent';
