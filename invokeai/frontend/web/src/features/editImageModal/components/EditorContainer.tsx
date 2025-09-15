import {
  Button,
  ButtonGroup,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Select,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { CropBox } from 'features/editImageModal/lib/editor';
import { closeEditImageModal, type EditImageModalState } from 'features/editImageModal/store';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useUploadImageMutation } from 'services/api/endpoints/images';

type Props = {
  editor: EditImageModalState['editor'];
  onApplyCrop: EditImageModalState['onApplyCrop'];
  onReady: EditImageModalState['onReady'];
};

const CROP_ASPECT_RATIO_MAP: Record<string, number> = {
  '16:9': 16 / 9,
  '3:2': 3 / 2,
  '4:3': 4 / 3,
  '1:1': 1,
  '3:4': 3 / 4,
  '2:3': 2 / 3,
  '9:16': 9 / 16,
};

const getAspectRatioString = (ratio: number | null) => {
  if (!ratio) {
    return 'free';
  }
  const entries = Object.entries(CROP_ASPECT_RATIO_MAP);
  for (const [key, value] of entries) {
    if (value === ratio) {
      return key;
    }
  }
  return 'free';
};

export const EditorContainer = ({ editor, onApplyCrop, onReady }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const [aspectRatio, setAspectRatio] = useState<string>('free');
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

  const [uploadImage] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });

  const setup = useCallback(
    async (container: HTMLDivElement) => {
      editor.init(container);
      editor.onZoomChange((zoom) => {
        setZoom(zoom);
      });
      editor.onCropBoxChange((crop) => {
        setCropBox(crop);
      });
      setAspectRatio(getAspectRatioString(editor.getCropAspectRatio()));
      await onReady();
      editor.fitToContainer();
    },
    [editor, onReady]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    setup(container);
    const handleResize = () => {
      editor.resize(container.clientWidth, container.clientHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    return () => {
      resizeObserver.disconnect();
    };
  }, [editor, setup]);

  const handleAspectRatioChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newRatio = e.target.value;
      setAspectRatio(newRatio);

      if (newRatio === 'free') {
        editor.setCropAspectRatio(null);
      } else {
        editor.setCropAspectRatio(CROP_ASPECT_RATIO_MAP[newRatio] ?? null);
      }
    },
    [editor]
  );

  const handleResetCrop = useCallback(() => {
    editor.resetCrop();
  }, [editor]);

  const handleApplyCrop = useCallback(async () => {
    await onApplyCrop();
    closeEditImageModal();
  }, [onApplyCrop]);

  const handleCancelCrop = useCallback(() => {
    closeEditImageModal();
  }, []);

  const handleExport = useCallback(async () => {
    try {
      const blob = await editor.exportImage('blob');
      const file = new File([blob], 'image.png', { type: 'image/png' });

      await uploadImage({
        file,
        is_intermediate: false,
        image_category: 'user',
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
      }).unwrap();
    } catch (err) {
      if (err instanceof Error && err.message.includes('tainted')) {
        alert(
          'Cannot export image: The image is from a different domain (CORS issue). To fix this:\n\n1. Load images from the same domain\n2. Use images from CORS-enabled sources\n3. Upload a local image file instead'
        );
      } else {
        alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }, [autoAddBoardId, editor, uploadImage]);

  const zoomIn = useCallback(() => {
    editor.zoomIn();
  }, [editor]);

  const zoomOut = useCallback(() => {
    editor.zoomOut();
  }, [editor]);

  const fitToContainer = useCallback(() => {
    editor.fitToContainer();
  }, [editor]);

  const resetView = useCallback(() => {
    editor.resetView();
  }, [editor]);

  return (
    <Flex w="full" h="full" flexDir="column" gap={4}>
      <Flex gap={2} alignItems="center">
        <FormControl flex={1}>
          <FormLabel>Aspect Ratio:</FormLabel>
          <Select size="sm" value={aspectRatio} onChange={handleAspectRatioChange} w={64}>
            <option value="free">Free</option>
            <option value="16:9">16:9</option>
            <option value="3:2">3:2</option>
            <option value="4:3">4:3</option>
            <option value="1:1">1:1 (Square)</option>
            <option value="3:4">3:4</option>
            <option value="2:3">2:3 (Portrait)</option>
            <option value="9:16">9:16 (Portrait)</option>
          </Select>
        </FormControl>

        <Spacer />

        <ButtonGroup size="sm" isAttached={false}>
          <Button onClick={fitToContainer}>Fit View</Button>
          <Button onClick={resetView}>Reset View</Button>
          <Button onClick={zoomIn}>Zoom In</Button>
          <Button onClick={zoomOut}>Zoom Out</Button>
        </ButtonGroup>

        <Spacer />

        <ButtonGroup size="sm" isAttached={false}>
          <Button onClick={handleApplyCrop}>Apply</Button>
          <Button onClick={handleResetCrop}>Reset</Button>
          <Button onClick={handleCancelCrop}>Cancel</Button>
          <Button onClick={handleExport}>Export</Button>
        </ButtonGroup>
      </Flex>

      <Flex position="relative" w="full" h="full" bg="base.900">
        <Flex position="absolute" inset={0} ref={containerRef} />
      </Flex>

      <Flex gap={2} color="base.300">
        <Text>Mouse wheel: Zoom</Text>
        <Divider orientation="vertical" />
        <Text>Space + Drag: Pan</Text>
        <Divider orientation="vertical" />
        <Text>Drag crop box or handles to adjust</Text>
        {cropBox && (
          <>
            <Divider orientation="vertical" />
            <Text>
              X: {Math.round(cropBox.x)}, Y: {Math.round(cropBox.y)}, Width: {Math.round(cropBox.width)}, Height:{' '}
              {Math.round(cropBox.height)}
            </Text>
          </>
        )}
        <Spacer key="help-spacer" />
        <Text key="help-zoom">Zoom: {Math.round(zoom * 100)}%</Text>
      </Flex>
    </Flex>
  );
};
