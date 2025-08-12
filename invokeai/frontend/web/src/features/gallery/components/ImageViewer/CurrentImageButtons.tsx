import { Button, Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
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
import { PostProcessingPopover } from 'features/parameters/components/PostProcessing/PostProcessingPopover';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
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
import type { ImageDTO } from 'services/api/types';

export const CurrentImageButtons = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const { t } = useTranslation();
  const tab = useAppSelector(selectActiveTab);
  const isCanvasOrGenerateTab = tab === 'canvas' || tab === 'generate';
  const isCanvasOrGenerateOrUpscalingTab = tab === 'canvas' || tab === 'generate' || tab === 'upscaling';

  const isUpscalingEnabled = useFeatureStatus('upscaling');

  const recallAll = useRecallAll(imageDTO);
  const recallRemix = useRecallRemix(imageDTO);
  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);
  const recallDimensions = useRecallDimensions(imageDTO);
  const loadWorkflow = useLoadWorkflow(imageDTO);
  const editImage = useEditImage(imageDTO);
  const deleteImage = useDeleteImage(imageDTO);

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

      <Button
        leftIcon={<PiPencilBold />}
        onClick={editImage.edit}
        isDisabled={!editImage.isEnabled}
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
        isDisabled={!loadWorkflow.isEnabled}
        variant="link"
        alignSelf="stretch"
        onClick={loadWorkflow.load}
      />
      {isCanvasOrGenerateTab && (
        <IconButton
          icon={<PiArrowsCounterClockwiseBold />}
          tooltip={`${t('parameters.remixImage')} (R)`}
          aria-label={`${t('parameters.remixImage')} (R)`}
          isDisabled={!recallRemix.isEnabled}
          variant="link"
          alignSelf="stretch"
          onClick={recallRemix.recall}
        />
      )}
      {isCanvasOrGenerateOrUpscalingTab && (
        <IconButton
          icon={<PiQuotesBold />}
          tooltip={`${t('parameters.usePrompt')} (P)`}
          aria-label={`${t('parameters.usePrompt')} (P)`}
          isDisabled={!recallPrompts.isEnabled}
          variant="link"
          alignSelf="stretch"
          onClick={recallPrompts.recall}
        />
      )}
      {isCanvasOrGenerateOrUpscalingTab && (
        <IconButton
          icon={<PiPlantBold />}
          tooltip={`${t('parameters.useSeed')} (S)`}
          aria-label={`${t('parameters.useSeed')} (S)`}
          isDisabled={!recallSeed.isEnabled}
          variant="link"
          alignSelf="stretch"
          onClick={recallSeed.recall}
        />
      )}
      {isCanvasOrGenerateTab && (
        <IconButton
          icon={<PiRulerBold />}
          tooltip={`${t('parameters.useSize')} (D)`}
          aria-label={`${t('parameters.useSize')} (D)`}
          variant="link"
          alignSelf="stretch"
          onClick={recallDimensions.recall}
          isDisabled={!recallDimensions.isEnabled}
        />
      )}
      {isCanvasOrGenerateTab && (
        <IconButton
          icon={<PiAsteriskBold />}
          tooltip={`${t('parameters.useAll')} (A)`}
          aria-label={`${t('parameters.useAll')} (A)`}
          isDisabled={!recallAll.isEnabled}
          variant="link"
          alignSelf="stretch"
          onClick={recallAll.recall}
        />
      )}

      {isUpscalingEnabled && <PostProcessingPopover imageDTO={imageDTO} isDisabled={false} />}

      <Divider orientation="vertical" h={8} mx={2} />

      <DeleteImageButton onClick={deleteImage.delete} isDisabled={!deleteImage.isEnabled} />
    </>
  );
});

CurrentImageButtons.displayName = 'CurrentImageButtons';
