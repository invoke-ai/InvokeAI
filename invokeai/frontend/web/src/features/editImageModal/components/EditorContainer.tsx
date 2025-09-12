import { Button, Divider, Flex, Select, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import type { CropBox, Editor } from 'features/editImageModal/lib/editor';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useGetImageDTOQuery, useUploadImageMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

type Props = {
  editor: Editor;
  imageName: string;
};

export const EditorContainer = ({ editor, imageName }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [cropInProgress, setCropInProgress] = useState(false);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const [cropApplied, setCropApplied] = useState(false);
  const [aspectRatio, setAspectRatio] = useState<string>('free');
  const { data: imageDTO } = useGetImageDTOQuery(imageName);
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

  const [uploadImage] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });

  const setup = useCallback(
    async (imageDTO: ImageDTO, container: HTMLDivElement) => {
      editor.init(container);
      editor.setCallbacks({
        onZoomChange: (zoom) => {
          setZoom(zoom);
        },
        onCropStart: () => {
          setCropInProgress(true);
          setCropBox(null);
        },
        onCropBoxChange: (crop) => {
          setCropBox(crop);
        },
        onCropApply: () => {
          setCropApplied(true);
          setCropInProgress(false);
          setCropBox(null);
        },
        onCropReset: () => {
          setCropApplied(true);
          setCropInProgress(false);
          setCropBox(null);
        },
        onCropCancel: () => {
          setCropInProgress(false);
          setCropBox(null);
        },
        onImageLoad: () => {
          // setCropInfo('');
          // setIsCropping(false);
          // setHasCropBbox(false);
        },
      });
      const blob = await convertImageUrlToBlob(imageDTO.image_url);
      if (!blob) {
        console.error('Failed to convert image to blob');
        return;
      }

      await editor.loadImage(imageDTO.image_url);
      editor.startCrop({
        x: 0,
        y: 0,
        width: imageDTO.width,
        height: imageDTO.height,
      });
    },
    [editor]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !imageDTO) {
      return;
    }
    editor.init(container);
    setup(imageDTO, container);
    const handleResize = () => {
      editor.resize(container.clientWidth, container.clientHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    return () => {
      resizeObserver.disconnect();
    };
  }, [editor, imageDTO, setup]);

  const handleStartCrop = useCallback(() => {
    editor.startCrop();
    // Apply current aspect ratio if not free
    if (aspectRatio !== 'free') {
      const ratios: Record<string, number> = {
        '1:1': 1,
        '4:3': 4 / 3,
        '16:9': 16 / 9,
        '3:2': 3 / 2,
        '2:3': 2 / 3,
        '9:16': 9 / 16,
      };
      editor.setCropAspectRatio(ratios[aspectRatio]);
    }
  }, [aspectRatio, editor]);

  const handleAspectRatioChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newRatio = e.target.value;
      setAspectRatio(newRatio);

      if (newRatio === 'free') {
        editor.setCropAspectRatio(undefined);
      } else {
        const ratios: Record<string, number> = {
          '1:1': 1,
          '4:3': 4 / 3,
          '16:9': 16 / 9,
          '3:2': 3 / 2,
          '2:3': 2 / 3,
          '9:16': 9 / 16,
        };
        editor.setCropAspectRatio(ratios[newRatio]);
      }
    },
    [editor]
  );

  const handleApplyCrop = useCallback(() => {
    editor.applyCrop();
    // setIsCropping(false);
    // setHasCropBbox(true);
    // setCropInfo('');
    setAspectRatio('free');
  }, [editor]);

  const handleCancelCrop = useCallback(() => {
    editor.cancelCrop();
    // setIsCropping(false);
    // setCropInfo('');
    setAspectRatio('free');
  }, [editor]);

  const handleResetCrop = useCallback(() => {
    editor.resetCrop();
    // setHasCropBbox(false);
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
