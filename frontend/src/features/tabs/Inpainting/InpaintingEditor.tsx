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
} from '@chakra-ui/react';

import {
  MouseEvent,
  MutableRefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { FaEraser, FaPaintBrush, FaTrash } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import {
  setInpaintingBrushRadius,
  setInpaintingTool,
} from '../../options/optionsSlice';
import InpaintingCanvas, {
  canvasBgImage,
  canvasRef,
  inpaintingOptionsSelector,
  maskCanvas,
} from './InpaintingCanvas';

type Tool = 'pen' | 'eraser';

type Point = {
  x: number;
  y: number;
};

function distanceBetween(point1: Point, point2: Point) {
  return Math.sqrt(
    Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
  );
}

function angleBetween(point1: Point, point2: Point) {
  return Math.atan2(point2.x - point1.x, point2.y - point1.y);
}

// export let canvasRef: MutableRefObject<HTMLCanvasElement | null>;
// const maskCanvas = document.createElement('canvas');
// const brushPreviewCanvas = document.createElement('canvas');

const InpaintingEditor = () => {
  const { inpaintingTool: tool, inpaintingBrushRadius: brushRadius } =
    useAppSelector(inpaintingOptionsSelector);

  const dispatch = useAppDispatch();

  // TODO: add mask invert display (so u can see exactly what parts of image are masked)
  const [shouldInvertMask, setShouldInvertMask] = useState<boolean>(false);

  // TODO: add mask overlay display
  const [shouldOverlayMask, setShouldOverlayMask] = useState<boolean>(false);

  // TODO: draw brush preview separately using this and cursorPos
  const [shouldShowBrushPreview, setShouldShowBrushPreview] =
    useState<boolean>(false);
  const [lastCursorPosition, setLastCursorPosition] = useState<Point>({
    x: 0,
    y: 0,
  });
  //

  const handleMouseOverBrushControls = () => {
    setShouldShowBrushPreview(true);
  };

  const handleMouseOutBrushControls = () => {
    setShouldShowBrushPreview(false);
  };

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
          <FormControl
            width={300}
            onMouseOver={handleMouseOverBrushControls}
            onMouseOut={handleMouseOutBrushControls}
          >
            <FormLabel>Brush Radius</FormLabel>

            <Slider
              aria-label="radius"
              value={brushRadius}
              onChange={(v: number) => {
                dispatch(setInpaintingBrushRadius(v));
              }}
              min={1}
              max={500}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderMark
                value={brushRadius}
                textAlign="center"
                bg="gray.800"
                color="white"
                mt="-10"
                ml="-5"
                w="12"
              >
                {brushRadius}px
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
      <div className="inpainting-wrapper checkerboard">
        <InpaintingCanvas />
      </div>
    </div>
  );
};

export default InpaintingEditor;
