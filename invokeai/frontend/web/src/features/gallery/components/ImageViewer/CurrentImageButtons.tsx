import { Button, Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useDeleteImage } from 'features/gallery/hooks/useDeleteImage';
import { useEditImage } from 'features/gallery/hooks/useEditImage';
import { useLoadWorkflow } from 'features/gallery/hooks/useLoadWorkflow';
import { useRecallAll } from 'features/gallery/hooks/useRecallAll';
import { useRecallDimensions } from 'features/gallery/hooks/useRecallDimensions';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallRemix } from 'features/gallery/hooks/useRecallRemix';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { PostProcessingPopover } from 'features/parameters/components/PostProcessing/PostProcessingPopover';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiDotsThreeOutlineFill,
  PiFlowArrowBold,
  PiPencilBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useImageDTO } from 'services/api/endpoints/images';

import { useImageViewerContext } from './context';

export const CurrentImageButtons = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageViewerContext();
  const hasProgressImage = useStore(ctx.$hasProgressImage);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const isDisabledOverride = hasProgressImage && shouldShowProgressInViewer;

  const imageName = useAppSelector(selectLastSelectedImage);
  const imageDTO = useImageDTO(imageName);

  const isUpscalingEnabled = useFeatureStatus('upscaling');

  const recallAll = useRecallAll(imageDTO);
  const recallRemix = useRecallRemix(imageDTO);
  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);
  const recallDimensions = useRecallDimensions(imageDTO);
  const loadWorkflow = useLoadWorkflow(imageDTO);
  const editImage = useEditImage(imageDTO);
  const deleteImage = useDeleteImage(imageDTO);

  console.log(isDisabledOverride, recallSeed.isEnabled);

  return (
    <>
      <Menu isLazy>
        <MenuButton
          as={IconButton}
          aria-label={t('parameters.imageActions')}
          tooltip={t('parameters.imageActions')}
          isDisabled={isDisabledOverride || !imageDTO}
          variant="link"
          alignSelf="stretch"
          icon={<PiDotsThreeOutlineFill />}
        />
        <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
      </Menu>

      <Divider orientation="vertical" h={8} mx={2} />

      <Button
        leftIcon={<PiPencilBold />}
        onClick={editImage.edit}
        isDisabled={isDisabledOverride || !editImage.isEnabled}
        variant="link"
        size="sm"
        alignSelf="stretch"
        px={2}
      >
        {t('common.edit')}
      </Button>

      <Divider orientation="vertical" h={8} mx={2} />

      <IconButton
        icon={<PiFlowArrowBold />}
        tooltip={`${t('nodes.loadWorkflow')} (W)`}
        aria-label={`${t('nodes.loadWorkflow')} (W)`}
        isDisabled={isDisabledOverride || !loadWorkflow.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={loadWorkflow.load}
      />
      <IconButton
        icon={<PiArrowsCounterClockwiseBold />}
        tooltip={`${t('parameters.remixImage')} (R)`}
        aria-label={`${t('parameters.remixImage')} (R)`}
        isDisabled={isDisabledOverride || !recallRemix.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={recallRemix.recall}
      />
      <IconButton
        icon={<PiQuotesBold />}
        tooltip={`${t('parameters.usePrompt')} (P)`}
        aria-label={`${t('parameters.usePrompt')} (P)`}
        isDisabled={isDisabledOverride || !recallPrompts.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={recallPrompts.recall}
      />
      <IconButton
        icon={<PiPlantBold />}
        tooltip={`${t('parameters.useSeed')} (S)`}
        aria-label={`${t('parameters.useSeed')} (S)`}
        isDisabled={isDisabledOverride || !recallSeed.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={recallSeed.recall}
      />
      <IconButton
        icon={<PiRulerBold />}
        tooltip={`${t('parameters.useSize')} (D)`}
        aria-label={`${t('parameters.useSize')} (D)`}
        variant="link"
        alignSelf="stretch"
        onClick={recallDimensions.recall}
        isDisabled={isDisabledOverride || !recallDimensions.isEnabled}
      />
      <IconButton
        icon={<PiAsteriskBold />}
        tooltip={`${t('parameters.useAll')} (A)`}
        aria-label={`${t('parameters.useAll')} (A)`}
        isDisabled={isDisabledOverride || !recallAll.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={recallAll.recall}
      />

      {isUpscalingEnabled && <PostProcessingPopover imageDTO={imageDTO} isDisabled={isDisabledOverride} />}

      <Divider orientation="vertical" h={8} mx={2} />

      <DeleteImageButton onClick={deleteImage.delete} isDisabled={isDisabledOverride || !deleteImage.isEnabled} />
    </>
  );
});

CurrentImageButtons.displayName = 'CurrentImageButtons';
