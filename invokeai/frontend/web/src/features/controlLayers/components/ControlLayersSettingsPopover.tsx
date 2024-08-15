import {
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { MaskOpacity } from 'features/controlLayers/components/MaskOpacity';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import {
  clipToBboxChanged,
  invertScrollChanged,
  rasterizationCachesInvalidated,
} from 'features/controlLayers/store/canvasV2Slice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

const ControlLayersSettingsPopover = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useStore($canvasManager);
  const clipToBbox = useAppSelector((s) => s.canvasV2.settings.clipToBbox);
  const invertScroll = useAppSelector((s) => s.canvasV2.tool.invertScroll);
  const onChangeInvertScroll = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(invertScrollChanged(e.target.checked)),
    [dispatch]
  );
  const onChangeClipToBbox = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(clipToBboxChanged(e.target.checked)),
    [dispatch]
  );
  const invalidateRasterizationCaches = useCallback(() => {
    dispatch(rasterizationCachesInvalidated());
  }, [dispatch]);
  const calculateBboxes = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    for (const adapter of canvasManager.rasterLayerAdapters.values()) {
      adapter.transformer.requestRectCalculation();
    }
    for (const adapter of canvasManager.controlLayerAdapters.values()) {
      adapter.transformer.requestRectCalculation();
    }
    for (const adapter of canvasManager.regionalGuidanceAdapters.values()) {
      adapter.transformer.requestRectCalculation();
    }
    canvasManager.inpaintMaskAdapter.transformer.requestRectCalculation();
  }, [canvasManager]);
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('common.settingsLabel')} icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <MaskOpacity />
            <FormControl w="full">
              <FormLabel flexGrow={1}>{t('unifiedCanvas.invertBrushSizeScrollDirection')}</FormLabel>
              <Checkbox isChecked={invertScroll} onChange={onChangeInvertScroll} />
            </FormControl>
            <FormControl w="full">
              <FormLabel flexGrow={1}>{t('unifiedCanvas.clipToBbox')}</FormLabel>
              <Checkbox isChecked={clipToBbox} onChange={onChangeClipToBbox} />
            </FormControl>
            <Button onClick={invalidateRasterizationCaches} size="sm">
              Invalidate Rasterization Caches
            </Button>
            <Button onClick={calculateBboxes} size="sm">
              Calculate Bboxes
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ControlLayersSettingsPopover);
