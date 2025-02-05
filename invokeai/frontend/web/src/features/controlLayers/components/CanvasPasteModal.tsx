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
          const { x, y } =
            destination === 'canvas'
              ? canvasManager.compositor.getVisibleRectOfType('raster_layer')
              : canvasManager.stateApi.getBbox().rect;

          createNewCanvasEntityFromImage({
            type: 'raster_layer',
            imageDTO,
            dispatch,
            getState,
            overrides: { position: { x, y } },
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
                ? t('gallery.galleryAssets')
                : destination === 'canvas'
                  ? t('controlLayers.canvasOrigin')
                  : t('controlLayers.canvasBbox'),
          }),
          status: 'success',
        });
      }
    },
    [autoAddBoardId, canvasManager.compositor, canvasManager.stateApi, dispatch, getState, t, uploadImage]
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
        <ModalHeader>{t('gallery.pasteTo')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Flex w="full" gap={2} justifyContent="center" alignItems="center">
            <Button onClick={pasteToAssets} isDisabled={isLoading}>
              {t('gallery.galleryAssets')}
            </Button>
            <Button onClick={pasteToBbox} isDisabled={isLoading}>
              {t('controlLayers.canvasBbox')}
            </Button>
            <Button onClick={pasteToCanvas} isDisabled={isLoading}>
              {t('controlLayers.canvasOrigin')}
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
