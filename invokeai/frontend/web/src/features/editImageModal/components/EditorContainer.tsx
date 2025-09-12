import { Button, Flex, Select } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import { useEditor } from 'features/editImageModal/hooks/useEditor';
import { $imageName } from 'features/editImageModal/store';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useGetImageDTOQuery, useUploadImageMutation } from 'services/api/endpoints/images';

export const EditorContainer = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const editor = useEditor({ containerRef });
  const [zoomLevel, setZoomLevel] = useState(100);
  const [cropInfo, setCropInfo] = useState<string>('');
  const [isCropping, setIsCropping] = useState(false);
  const [hasCropBbox, setHasCropBbox] = useState(false);
  const [aspectRatio, setAspectRatio] = useState<string>('free');
  const { data: imageDTO } = useGetImageDTOQuery($imageName.get() ?? skipToken);
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

  const [uploadImage, { isLoading }] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });

  const loadImage = useCallback(async () => {
    if (!imageDTO) {
      console.error('Image not found');
      return;
    }
    const blob = await convertImageUrlToBlob(imageDTO.image_url);
    if (!blob) {
      console.error('Failed to convert image to blob');
      return;
    }
    await editor.loadImage(blob);
  }, [editor, imageDTO]);

  // Setup callbacks
  useEffect(() => {
    loadImage();
    editor.setCallbacks({
      onZoomChange: (zoom) => setZoomLevel(Math.round(zoom * 100)),
      onCropChange: (crop) => {
        if (!crop) {
          setCropInfo('');
          return;
        }
        setCropInfo(
          `Crop: ${Math.round(crop.x)}, ${Math.round(crop.y)} - ${Math.round(crop.width)}x${Math.round(crop.height)}`
        );
      },
      onImageLoad: () => {
        setCropInfo('');
        setIsCropping(false);
        setHasCropBbox(false);
      },
    });
  }, [editor, loadImage]);

  const handleStartCrop = useCallback(() => {
    editor.startCrop();
    setIsCropping(true);
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
    setIsCropping(false);
    setHasCropBbox(true);
    setCropInfo('');
    setAspectRatio('free');
  }, [editor]);

  const handleCancelCrop = useCallback(() => {
    editor.cancelCrop();
    setIsCropping(false);
    setCropInfo('');
    setAspectRatio('free');
  }, [editor]);

  const handleResetCrop = useCallback(() => {
    editor.resetCrop();
    setHasCropBbox(false);
  }, [editor]);

  const handleExport = useCallback(async () => {
    try {
      const blob = (await editor.exportImage('blob')) as Blob;
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
    <Flex w="full" h="full" flexDir="column">
      <Flex>
        <Flex>
          {!isCropping && (
            <>
              <Button onClick={handleStartCrop}>Start Crop</Button>
              {hasCropBbox && <Button onClick={handleResetCrop}>Reset Crop</Button>}
            </>
          )}
          {isCropping && (
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
        </Flex>

        <Flex>
          <Button onClick={fitToContainer}>Fit</Button>
          <Button onClick={resetView}>Reset View</Button>
          <Button onClick={zoomIn}>Zoom In</Button>
          <Button onClick={zoomOut}>Zoom Out</Button>
        </Flex>

        <Button onClick={handleExport}>Export</Button>

        <Flex>
          <span>Zoom: {zoomLevel}%</span>
          {cropInfo && <span>{cropInfo}</span>}
          {hasCropBbox && <span style={{ color: 'green' }}>✓ Crop Applied</span>}
        </Flex>
      </Flex>

      <Flex>
        <Flex>Mouse wheel: Zoom</Flex>
        <Flex>Space + Drag: Pan</Flex>
        {isCropping && <Flex>Drag crop box or handles to adjust</Flex>}
        {isCropping && aspectRatio !== 'free' && <Flex>Aspect ratio: {aspectRatio}</Flex>}
      </Flex>
      <Flex position="relative" w="full" h="full" bg="base.900">
        <Flex position="absolute" inset={0} ref={containerRef} />
      </Flex>
    </Flex>
  );
};
