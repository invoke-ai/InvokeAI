import { Button, Divider, Flex, Select, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { CropBox, Editor } from 'features/editImageModal/lib/editor';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useUploadImageMutation } from 'services/api/endpoints/images';

type Props = {
  editor: Editor;
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

export const getAspectRatioString = (ratio: number | null) => {
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

export const EditorContainer = ({ editor }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [cropInProgress, setCropInProgress] = useState(false);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const [cropApplied, setCropApplied] = useState(false);
  const [aspectRatio, setAspectRatio] = useState<string>('free');
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

  const [uploadImage] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });

  const setup = useCallback(
    (container: HTMLDivElement) => {
      editor.init(container);
      editor.onZoomChange((zoom) => {
        setZoom(zoom);
      });
      editor.onCropStart(() => {
        setCropInProgress(true);
        setCropBox(null);
      });
      editor.onCropBoxChange((crop) => {
        setCropBox(crop);
      });
      editor.onCropApply(() => {
        setCropApplied(true);
        setCropInProgress(false);
        setCropBox(null);
      });
      editor.onCropReset(() => {
        setCropApplied(true);
        setCropInProgress(false);
        setCropBox(null);
      });
      editor.onCropCancel(() => {
        setCropInProgress(false);
        setCropBox(null);
      });
      editor.onImageLoad(() => {
        // setCropInfo('');
        // setIsCropping(false);
        // setHasCropBbox(false);
      });
      setAspectRatio(getAspectRatioString(editor.getCropAspectRatio()));
      editor.fitToContainer();
    },
    [editor]
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

  const handleStartCrop = useCallback(() => {
    editor.startCrop();
    // Apply current aspect ratio if not free
    if (aspectRatio !== 'free') {
      editor.setCropAspectRatio(CROP_ASPECT_RATIO_MAP[aspectRatio] ?? null);
    }
  }, [aspectRatio, editor]);

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

  const handleApplyCrop = useCallback(() => {
    editor.applyCrop();
  }, [editor]);

  const handleCancelCrop = useCallback(() => {
    editor.cancelCrop();
  }, [editor]);

  const handleResetCrop = useCallback(() => {
    editor.resetCrop();
  }, [editor]);

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
      console.error('Export failed:', err);
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
      <Flex gap={2}>
        {!cropInProgress && <Button onClick={handleStartCrop}>Start Crop</Button>}
        {cropApplied && <Button onClick={handleResetCrop}>Reset Crop</Button>}
        {cropInProgress && (
          <>
            <Select value={aspectRatio} onChange={handleAspectRatioChange}>
              <option value="free">Free</option>
              <option value="1:1">1:1 (Square)</option>
              <option value="4:3">4:3</option>
              <option value="16:9">16:9</option>
              <option value="3:2">3:2</option>
              <option value="2:3">2:3 (Portrait)</option>
              <option value="9:16">9:16 (Portrait)</option>
            </Select>
            <Button onClick={handleApplyCrop}>Apply Crop</Button>
            <Button onClick={handleCancelCrop}>Cancel Crop</Button>
          </>
        )}

        <Button onClick={fitToContainer}>Fit</Button>
        <Button onClick={resetView}>Reset View</Button>
        <Button onClick={zoomIn}>Zoom In</Button>
        <Button onClick={zoomOut}>Zoom Out</Button>

        <Button onClick={handleExport}>Export</Button>
      </Flex>

      <Flex position="relative" w="full" h="full" bg="base.900">
        <Flex position="absolute" inset={0} ref={containerRef} />
      </Flex>

      <Flex gap={2} color="base.300">
        <Text>Mouse wheel: Zoom</Text>
        <Divider orientation="vertical" />
        <Text>Space + Drag: Pan</Text>
        {cropInProgress && (
          <>
            <Divider orientation="vertical" />
            <Text>Drag crop box or handles to adjust</Text>
          </>
        )}
        {cropInProgress && cropBox && (
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
