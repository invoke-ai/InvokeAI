import { Button, Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDeleteVideo } from 'features/deleteImageModal/hooks/use-delete-video';
import { DeleteVideoButton } from 'features/deleteVideoModal/components/DeleteVideoButton';
import SingleSelectionVideoMenuItems from 'features/gallery/components/ContextMenu/SingleSelectionVideoMenuItems';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useGalleryPanel } from 'features/ui/layouts/use-gallery-panel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import { useCaptureVideoFrame } from 'features/video/hooks/useCaptureVideoFrame';
import { memo, useCallback, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCameraBold, PiCrosshairBold, PiDotsThreeOutlineFill, PiSpinnerBold } from 'react-icons/pi';
import type { VideoDTO } from 'services/api/types';

export const CurrentVideoButtons = memo(({ videoDTO }: { videoDTO: VideoDTO }) => {
  const { t } = useTranslation();
  const tab = useAppSelector(selectActiveTab);
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const galleryPanel = useGalleryPanel(activeTab);
  const deleteVideo = useDeleteVideo(videoDTO);

  const captureVideoFrame = useCaptureVideoFrame();
  const { videoRef } = useVideoViewerContext();
  const [capturing, setCapturing] = useState(false);

  const locateInGallery = useCallback(() => {
    navigationApi.expandRightPanel();
    galleryPanel.expand();
    flushSync(() => {
      dispatch(
        boardIdSelected({
          boardId: videoDTO.board_id ?? 'none',
          select: {
            selection: [{ type: 'video', id: videoDTO.video_id }],
            galleryView: 'videos',
          },
        })
      );
    });
  }, [dispatch, galleryPanel, videoDTO]);

  const onClickSaveFrame = useCallback(async () => {
    setCapturing(true);
    await captureVideoFrame(videoRef.current);
    setCapturing(false);
  }, [captureVideoFrame, videoRef]);

  const doesTabHaveGallery = tab === 'canvas' || tab === 'generate' || tab === 'workflows' || tab === 'upscaling';

  // const recallAll = useRecallAll(imageDTO);
  // const recallRemix = useRecallRemix(imageDTO);
  // const recallPrompts = useRecallPrompts(imageDTO);
  // const recallSeed = useRecallSeed(imageDTO);
  // const recallDimensions = useRecallDimensions(imageDTO);
  // const loadWorkflow = useLoadWorkflow(imageDTO);
  // const editImage = useEditImage(imageDTO);
  // const deleteImage = useDeleteImage(imageDTO);

  return (
    <>
      <Menu isLazy>
        <MenuButton
          as={IconButton}
          aria-label={t('parameters.videoActions')}
          tooltip={t('parameters.videoActions')}
          isDisabled={!videoDTO}
          variant="link"
          alignSelf="stretch"
          icon={<PiDotsThreeOutlineFill />}
        />
        <MenuList>{videoDTO && <SingleSelectionVideoMenuItems videoDTO={videoDTO} />}</MenuList>
      </Menu>

      <Divider orientation="vertical" h={8} mx={2} />

      <Button
        leftIcon={capturing ? <PiSpinnerBold /> : <PiCameraBold />}
        onClick={onClickSaveFrame}
        isDisabled={capturing || !videoRef}
        variant="link"
        size="sm"
        alignSelf="stretch"
        px={2}
        isLoading={capturing}
        loadingText="Capturing..."
      >
        {capturing ? 'Capturing...' : 'Save Current Frame'}
      </Button>

      <Divider orientation="vertical" h={8} mx={2} />

      {doesTabHaveGallery && (
        <>
          <IconButton
            icon={<PiCrosshairBold />}
            aria-label={t('boards.locateInGalery')}
            tooltip={t('boards.locateInGalery')}
            onClick={locateInGallery}
            variant="link"
            size="sm"
            alignSelf="stretch"
          />
          <Divider orientation="vertical" h={8} mx={2} />
        </>
      )}

      <DeleteVideoButton onClick={deleteVideo.delete} isDisabled={!deleteVideo.isEnabled} />
    </>
  );
});

CurrentVideoButtons.displayName = 'CurrentVideoButtons';
