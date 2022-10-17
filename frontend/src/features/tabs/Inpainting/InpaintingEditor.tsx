import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Switch,
  Flex,
  FormControl,
  FormLabel,
  SliderMark,
  useToast,
} from '@chakra-ui/react';

import { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaEraser, FaPaintBrush, FaTrash } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import {
  setInpaintingBrushSize,
  setInpaintingTool,
} from '../../options/optionsSlice';
import InpaintingCanvas, {
  canvasBgImage,
  canvasRef,
  inpaintingOptionsSelector,
  maskCanvas,
} from './InpaintingCanvas';

const InpaintingEditor = () => {
  const { tool, brushSize, activeTab } = useAppSelector(
    inpaintingOptionsSelector
  );

  const dispatch = useAppDispatch();
  const toast = useToast();

  // TODO: add mask invert display (so u can see exactly what parts of image are masked)
  const [shouldInvertMask, setShouldInvertMask] = useState<boolean>(false);

  // TODO: add mask overlay display
  const [shouldOverlayMask, setShouldOverlayMask] = useState<boolean>(false);

  // Hotkeys
  useHotkeys(
    '[',
    () => {
      if (activeTab === 'inpainting' && brushSize - 5 > 0) {
        dispatch(setInpaintingBrushSize(brushSize - 5));
      } else {
        dispatch(setInpaintingBrushSize(1));
      }
    },
    [brushSize]
  );

  useHotkeys(
    ']',
    () => {
      if (activeTab === 'inpainting') {
        dispatch(setInpaintingBrushSize(brushSize + 5));
      }
    },
    [brushSize]
  );

  useHotkeys('b', () => {
    if (activeTab === 'inpainting') {
      dispatch(setInpaintingTool('eraser'));
    }
  });

  useHotkeys('e', () => {
    if (activeTab === 'inpainting') {
      dispatch(setInpaintingTool('uneraser'));
    }
  });

  useHotkeys('c', () => {
    if (activeTab === 'inpainting') {
      handleClickClearMask();
      toast({
        title: 'Mask Cleared',
        status: 'success',
        duration: 2500,
        isClosable: true,
      });
    }
  });
  const handleClickClearMask = () => {
    if (!canvasRef.current) return;
    const canvasContext = canvasRef.current.getContext('2d');
    const maskCanvasContext = maskCanvas.getContext('2d');

    if (!canvasContext || !canvasBgImage?.current || !maskCanvasContext) return;

    canvasContext.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );
    maskCanvasContext.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    // composite & draw the image
    canvasContext.globalCompositeOperation = 'source-over';
    canvasContext.drawImage(canvasBgImage.current, 0, 0);
  };

  return (
    <div>
      <Flex gap={4} direction={'column'} padding={2}>
        <Flex gap={4}>
          <IAIIconButton
            aria-label="Eraser"
            tooltip="Eraser"
            icon={<FaEraser />}
            colorScheme={tool === 'eraser' ? 'green' : undefined}
            onClick={() => dispatch(setInpaintingTool('eraser'))}
          />
          <IAIIconButton
            aria-label="Un-eraser"
            tooltip="Un-eraser"
            icon={<FaPaintBrush />}
            colorScheme={tool === 'uneraser' ? 'green' : undefined}
            onClick={() => dispatch(setInpaintingTool('uneraser'))}
          />
          <IAIIconButton
            aria-label="Clear mask"
            tooltip="Clear mask"
            icon={<FaTrash />}
            colorScheme={'red'}
            onClick={handleClickClearMask}
          />
        </Flex>
        <Flex gap={4}>
          <FormControl width={300}>
            <FormLabel>Brush Radius</FormLabel>

            <Slider
              aria-label="radius"
              value={brushSize}
              onChange={(v: number) => {
                dispatch(setInpaintingBrushSize(v));
              }}
              min={1}
              max={500}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderMark
                value={brushSize}
                textAlign="center"
                bg="gray.800"
                color="white"
                mt="-10"
                ml="-5"
                w="12"
              >
                {brushSize}px
              </SliderMark>
              <SliderThumb />
            </Slider>
          </FormControl>
          <FormControl width={300}>
            <FormLabel>Invert Mask Display</FormLabel>

            <Switch
              isDisabled={true}
              checked={shouldInvertMask}
              onChange={(e) => setShouldInvertMask(e.target.checked)}
            />
          </FormControl>
        </Flex>
      </Flex>
      <InpaintingCanvas />
    </div>
  );
};

export default InpaintingEditor;
