import { Button, Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import SingleSelectionVideoMenuItems from 'features/gallery/components/ContextMenu/SingleSelectionVideoMenuItems';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { useGalleryPanel } from 'features/ui/layouts/use-gallery-panel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import { useCaptureVideoFrame } from 'features/video/hooks/useCaptureVideoFrame';
import { memo, useCallback, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCameraBold, PiCrosshairBold, PiDotsThreeOutlineFill, PiSpinnerBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';
import { useDeleteVideosMutation } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

const log = logger('video');

export const CurrentVideoButtons = memo(({ videoDTO }: { videoDTO: VideoDTO }) => {
  const { t } = useTranslation();
  const tab = useAppSelector(selectActiveTab);
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const galleryPanel = useGalleryPanel(activeTab);
  const [deleteVideos] = useDeleteVideosMutation();

  // Video frame capture functionality
  const { $videoRef } = useVideoViewerContext();
  const videoRef = useStore($videoRef);
  const { captureFrame } = useCaptureVideoFrame();
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

  const handleDelete = useCallback(() => {
    deleteVideos({ video_ids: [videoDTO.video_id] });
  }, [deleteVideos, videoDTO]);

  const onClickSaveFrame = useCallback(async () => {
    setCapturing(true);
    let file: File;
    try {
      if (!videoRef) {
        toast({
          status: 'error',
          title: 'Video not ready',
          description: 'Please wait for the video to load before capturing a frame.',
        });
        return;
      }

      file = captureFrame(videoRef);
      await uploadImage({ file, image_category: 'user', is_intermediate: false, silent: true });
      toast({
        status: 'success',
        title: 'Frame saved to assets tab',
      });
    } catch (error) {
      log.error({ error: serializeError(error as Error) }, 'Failed to capture frame');
      toast({
        status: 'error',
        title: 'Failed to capture frame',
        description: 'There was an error capturing the current video frame.',
      });
    } finally {
      setCapturing(false);
    }
  }, [captureFrame, videoRef]);

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
          aria-label={t('parameters.imageActions')}
          tooltip={t('parameters.imageActions')}
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
        <IconButton
          icon={<PiCrosshairBold />}
          aria-label={t('boards.locateInGalery')}
          tooltip={t('boards.locateInGalery')}
          onClick={locateInGallery}
          variant="link"
          size="sm"
          alignSelf="stretch"
        />
      )}

      <Divider orientation="vertical" h={8} mx={2} />

      <DeleteImageButton onClick={handleDelete} />
    </>
  );
});

CurrentVideoButtons.displayName = 'CurrentVideoButtons';
