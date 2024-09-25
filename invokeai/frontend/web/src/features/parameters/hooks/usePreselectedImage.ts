import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { withResultAsync } from 'common/util/result';
import { canvasReset } from 'features/controlLayers/store/actions';
import { settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { selectBboxRect } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { getImageDTO, getImageMetadata } from 'services/api/endpoints/images';

export type UsePreselectedImageArg = { imageName: string; action: 'sendToCanvas' | 'useAllParameters' };

export const usePreselectedImage = (arg?: UsePreselectedImageArg) => {
  const didUseRef = useRef(false);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const bboxRect = useAppSelector(selectBboxRect);
  const imageViewer = useImageViewer();

  const handleSendToCanvas = useCallback(
    async (imageName: string) => {
      const getImageDTOResult = await withResultAsync(() => getImageDTO(imageName));
      if (getImageDTOResult.isErr()) {
        toast({
          id: 'SENT_TO_CANVAS',
          title: t('toast.sentToUnifiedCanvas'),
          status: 'info',
        });
        return;
      }
      const imageDTO = getImageDTOResult.value;
      const imageObject = imageDTOToImageObject(imageDTO);
      const overrides: Partial<CanvasRasterLayerState> = {
        position: { x: bboxRect.x, y: bboxRect.y },
        objects: [imageObject],
      };
      dispatch(canvasReset());
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      dispatch(settingsSendToCanvasChanged(true));
      dispatch(setActiveTab('canvas'));
      dispatch(sentImageToCanvas());
      imageViewer.close();
      toast({
        id: 'SENT_TO_CANVAS',
        title: t('toast.sentToUnifiedCanvas'),
        status: 'info',
      });
    },
    [dispatch, bboxRect, imageViewer, t]
  );

  const handleUseAllMetadata = useCallback(async (imageName: string) => {
    const getImageMetadataResult = await withResultAsync(() => getImageMetadata(imageName));
    if (getImageMetadataResult.isErr()) {
      return;
    }
    const metadata = getImageMetadataResult.value;
    parseAndRecallAllMetadata(metadata, true);
  }, []);

  useEffect(() => {
    if (didUseRef.current || !arg) {
      return;
    }

    didUseRef.current = true;

    if (arg.action === 'sendToCanvas') {
      handleSendToCanvas(arg.imageName);
    } else if (arg.action === 'useAllParameters') {
      handleUseAllMetadata(arg.imageName);
    }
  }, [handleSendToCanvas, handleUseAllMetadata, arg]);
};
