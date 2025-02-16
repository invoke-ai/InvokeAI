import {
  Button,
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiImageBold } from 'react-icons/pi';
import { useUploadImageMutation } from 'services/api/endpoints/images';

const $imageFile = atom<File | null>(null);
export const setFileToPaste = (file: File) => $imageFile.set(file);
const clearFileToPaste = () => $imageFile.set(null);

export const CanvasPasteModal = memo(() => {
  useAssertSingleton('CanvasPasteModal');
  const { dispatch, getState } = useAppStore();
  const { t } = useTranslation();
  const imageToPaste = useStore($imageFile);
  const canvasManager = useCanvasManager();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const [uploadImage, { isLoading }] = useUploadImageMutation({ fixedCacheKey: 'canvasPasteModal' });

  const getPosition = useCallback(
    (destination: 'canvas' | 'bbox') => {
      const { x, y } = canvasManager.stateApi.getBbox().rect;
      if (destination === 'bbox') {
        return { x, y };
      }
      const rasterLayerAdapters = canvasManager.compositor.getVisibleAdaptersOfType('raster_layer');
      if (rasterLayerAdapters.length === 0) {
        return { x, y };
      }
      {
        const { x, y } = canvasManager.compositor.getRectOfAdapters(rasterLayerAdapters);
        return { x, y };
      }
    },
    [canvasManager.compositor, canvasManager.stateApi]
  );

  const handlePaste = useCallback(
    async (file: File, destination: 'assets' | 'canvas' | 'bbox') => {
      try {
        const is_intermediate = destination !== 'assets';
        const imageDTO = await uploadImage({
          file,
          is_intermediate,
          image_category: 'user',
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        }).unwrap();

        if (destination !== 'assets') {
          createNewCanvasEntityFromImage({
            type: 'raster_layer',
            imageDTO,
            dispatch,
            getState,
            overrides: { position: getPosition(destination) },
          });
        }
      } catch {
        toast({
          title: t('toast.pasteFailed'),
          status: 'error',
        });
      } finally {
        clearFileToPaste();
        toast({
          title: t('toast.pasteSuccess', {
            destination:
              destination === 'assets'
                ? t('controlLayers.pasteToAssets')
                : destination === 'bbox'
                  ? t('controlLayers.pasteToBbox')
                  : t('controlLayers.pasteToCanvas'),
          }),
          status: 'success',
        });
      }
    },
    [autoAddBoardId, dispatch, getPosition, getState, t, uploadImage]
  );

  const pasteToAssets = useCallback(() => {
    if (!imageToPaste) {
      return;
    }
    handlePaste(imageToPaste, 'assets');
  }, [handlePaste, imageToPaste]);

  const pasteToCanvas = useCallback(() => {
    if (!imageToPaste) {
      return;
    }
    handlePaste(imageToPaste, 'canvas');
  }, [handlePaste, imageToPaste]);

  const pasteToBbox = useCallback(() => {
    if (!imageToPaste) {
      return;
    }
    handlePaste(imageToPaste, 'bbox');
  }, [handlePaste, imageToPaste]);

  return (
    <Modal isOpen={imageToPaste !== null} onClose={clearFileToPaste} useInert={false} isCentered size="2xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{t('controlLayers.pasteTo')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" justifyContent="center">
          <Flex flexDir="column" gap={4} w="min-content">
            <Button size="lg" onClick={pasteToCanvas} isDisabled={isLoading} leftIcon={<PiImageBold />}>
              {t('controlLayers.pasteToCanvasDesc')}
            </Button>
            <Button size="lg" onClick={pasteToBbox} isDisabled={isLoading} leftIcon={<PiBoundingBoxBold />}>
              {t('controlLayers.pasteToBboxDesc')}
            </Button>
            <Button size="lg" onClick={pasteToAssets} isDisabled={isLoading} variant="ghost">
              {t('controlLayers.pasteToAssetsDesc')}
            </Button>
          </Flex>
        </ModalBody>
        <ModalFooter>
          <Button onClick={clearFileToPaste} variant="ghost" isLoading={isLoading}>
            {t('common.cancel')}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});

CanvasPasteModal.displayName = 'CanvasPasteModal';
