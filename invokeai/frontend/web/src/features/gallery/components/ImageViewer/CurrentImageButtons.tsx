import { ButtonGroup, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { iiLayerAdded } from 'features/controlLayers/store/controlLayersSlice';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { sentImageToImg2Img } from 'features/gallery/store/actions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { parseAndRecallImageDimensions } from 'features/metadata/util/handlers';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { size } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
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

const selectShouldDisableToolbarButtons = createSelector(
  selectSystemSlice,
  selectLastSelectedImage,
  (system, lastSelectedImage) => {
    const hasProgressImage = Boolean(system.denoiseProgress?.progress_image);
    return hasProgressImage || !lastSelectedImage;
  }
);

const CurrentImageButtons = () => {
  const dispatch = useAppDispatch();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const selection = useAppSelector((s) => s.gallery.selection);
  const shouldDisableToolbarButtons = useAppSelector(selectShouldDisableToolbarButtons);
  const templates = useStore($templates);
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  const { recallAll, remix, recallSeed, recallPrompts, hasMetadata, hasSeed, hasPrompts, isLoadingMetadata } =
    useImageActions(lastSelectedImage?.image_name);

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } = useGetAndLoadEmbeddedWorkflow({});

  const handleLoadWorkflow = useCallback(() => {
    if (!lastSelectedImage || !lastSelectedImage.has_workflow) {
      return;
    }
    getAndLoadEmbeddedWorkflow(lastSelectedImage.image_name);
  }, [getAndLoadEmbeddedWorkflow, lastSelectedImage]);

  useHotkeys('w', handleLoadWorkflow, [lastSelectedImage]);
  useHotkeys('a', recallAll, [recallAll]);
  useHotkeys('s', recallSeed, [recallSeed]);
  useHotkeys('p', recallPrompts, [recallPrompts]);
  useHotkeys('r', remix, [remix]);

  const handleUseSize = useCallback(() => {
    parseAndRecallImageDimensions(lastSelectedImage);
  }, [lastSelectedImage]);

  useHotkeys('d', handleUseSize, [handleUseSize]);

  const handleSendToImageToImage = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(sentImageToImg2Img());
    dispatch(iiLayerAdded(imageDTO));
    dispatch(setActiveTab('generation'));
  }, [dispatch, imageDTO]);

  useHotkeys('shift+i', handleSendToImageToImage, [imageDTO]);

  const handleClickUpscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(upscaleRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imagesToDeleteSelected(selection));
  }, [dispatch, imageDTO, selection]);

  useHotkeys(
    'delete',
    () => {
      handleDelete();
    },
    [dispatch, imageDTO]
  );

  return (
    <>
      <ButtonGroup isDisabled={shouldDisableToolbarButtons}>
        <Menu isLazy>
          <MenuButton
            as={IconButton}
            aria-label={t('parameters.imageActions')}
            tooltip={t('parameters.imageActions')}
            isDisabled={!imageDTO}
            icon={<PiDotsThreeOutlineFill />}
          />
          <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
        </Menu>
      </ButtonGroup>

      <ButtonGroup isDisabled={shouldDisableToolbarButtons}>
        <IconButton
          icon={<PiFlowArrowBold />}
          tooltip={`${t('nodes.loadWorkflow')} (W)`}
          aria-label={`${t('nodes.loadWorkflow')} (W)`}
          isDisabled={!imageDTO?.has_workflow || !size(templates)}
          onClick={handleLoadWorkflow}
          isLoading={getAndLoadEmbeddedWorkflowResult.isLoading}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiArrowsCounterClockwiseBold />}
          tooltip={`${t('parameters.remixImage')} (R)`}
          aria-label={`${t('parameters.remixImage')} (R)`}
          isDisabled={!hasMetadata}
          onClick={remix}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiQuotesBold />}
          tooltip={`${t('parameters.usePrompt')} (P)`}
          aria-label={`${t('parameters.usePrompt')} (P)`}
          isDisabled={!hasPrompts}
          onClick={recallPrompts}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiPlantBold />}
          tooltip={`${t('parameters.useSeed')} (S)`}
          aria-label={`${t('parameters.useSeed')} (S)`}
          isDisabled={!hasSeed}
          onClick={recallSeed}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiRulerBold />}
          tooltip={`${t('parameters.useSize')} (D)`}
          aria-label={`${t('parameters.useSize')} (D)`}
          onClick={handleUseSize}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiAsteriskBold />}
          tooltip={`${t('parameters.useAll')} (A)`}
          aria-label={`${t('parameters.useAll')} (A)`}
          isDisabled={!hasMetadata}
          onClick={recallAll}
        />
      </ButtonGroup>

      <ButtonGroup>
        <DeleteImageButton onClick={handleDelete} />
      </ButtonGroup>
    </>
  );
};

export default memo(CurrentImageButtons);
