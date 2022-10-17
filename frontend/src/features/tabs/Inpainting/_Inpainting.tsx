import {
  Button,
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
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import { useLayoutEffect, useState } from 'react';
import { FaEraser, FaPaintBrush, FaTrash } from 'react-icons/fa';
import { Circle, Layer, Line, Stage } from 'react-konva';
import { Image as KonvaImage } from 'react-konva';
import { RootState, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';

type Tool = 'pen' | 'eraser';

type Point = {
  x: number;
  y: number;
};

type MaskLine = {
  tool: Tool;
  strokeWidth: number;
  points: number[];
};

type MaskCircle = {
  tool: Tool;
  radius: number;
  x: number;
  y: number;
};

const Inpainting = () => {
  const currentImage = useAppSelector(
    (state: RootState) => state.gallery.currentImage
  );

  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [shouldInvertMask, setShouldInvertMask] = useState<boolean>(false);
  const [shouldShowBrushPreview, setShouldShowBrushPreview] =
    useState<boolean>(false);
  const [didMouseMove, setDidMouseMove] = useState<boolean>(false);
  const [cursorPos, setCursorPos] = useState<Point | null>(null);

  const [tool, setTool] = useState<'pen' | 'eraser'>('pen');
  const [lines, setLines] = useState<MaskLine[]>([]);
  const [circles, setCircles] = useState<MaskCircle[]>([]);
  const [brushRadius, setBrushRadius] = useState<number>(20);

  const [canvasBgImage, setCanvasBgImage] = useState<HTMLImageElement | null>(
    null
  );

  useLayoutEffect(() => {
    if (currentImage) {
      const image = new Image();
      image.src = currentImage.url;
      image.onload = () => {
        setCanvasBgImage(image);
      };
    }
  }, [currentImage]);

  const handleMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    setIsDrawing(true);
    const stage = e.target.getStage();
    if (stage) {
      const pos = stage.getPointerPosition();
      if (pos) {
        setLines([
          ...lines,
          {
            tool,
            strokeWidth: brushRadius,
            points: [pos.x, pos.y],
          },
        ]);
      }
    }
  };

  const handleMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    // no drawing - skipping
    const stage = e.target.getStage();
    const point = stage?.getPointerPosition();

    point && setCursorPos(point);

    if (!isDrawing) {
      return;
    }

    setDidMouseMove(true);
    if (point) {
      const lastLine = lines[lines.length - 1];
      // add point
      lastLine.points = lastLine.points.concat([point.x, point.y]);

      // replace last
      lines.splice(lines.length - 1, 1, lastLine);
      setLines(lines.concat());
    }
  };

  const handleMouseUp = (e: KonvaEventObject<MouseEvent>) => {
    setIsDrawing(false);

    if (!didMouseMove) {
      const stage = e.target.getStage();
      const point = stage?.getPointerPosition();
      point &&
        setCircles([
          ...circles,
          {
            tool,
            radius: brushRadius,
            x: point.x,
            y: point.y,
          },
        ]);
      console.log(circles);
    } else {
      setDidMouseMove(false);
    }
  };

  const handleMouseOutCanvas = () => {
    setCursorPos(null);
  };
  const handleMouseOverBrushControls = () => {
    setShouldShowBrushPreview(true);
  };

  const handleMouseOutBrushControls = () => {
    setShouldShowBrushPreview(false);
  };

  return (
    canvasBgImage && (
      <>
        <Flex gap={4} direction={'column'} padding={2}>
          <Flex gap={4}>
            <IAIIconButton
              aria-label="Eraser"
              tooltip="Eraser"
              icon={<FaEraser />}
              colorScheme={tool === 'pen' ? 'green' : undefined}
              onClick={() => setTool('pen')}
            />
            <IAIIconButton
              aria-label="Un-eraser"
              tooltip="Un-eraser"
              icon={<FaPaintBrush />}
              colorScheme={tool === 'eraser' ? 'green' : undefined}
              onClick={() => setTool('eraser')}
            />
            <IAIIconButton
              aria-label="Clear mask"
              tooltip="Clear mask"
              icon={<FaTrash />}
              colorScheme={'red'}
              onClick={() => {
                setLines([]), setCircles([]);
              }}
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
                  setBrushRadius(v);
                }}
                min={1}
                max={Math.floor(canvasBgImage.width / 2)}
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
                checked={shouldInvertMask}
                onChange={(e) => setShouldInvertMask(e.target.checked)}
              />
            </FormControl>
          </Flex>
        </Flex>
        <div className="inpainting-wrapper checkerboard">
          <Stage
            width={canvasBgImage.width}
            height={canvasBgImage.height}
            onMouseDown={handleMouseDown}
            onMousemove={handleMouseMove}
            onMouseup={handleMouseUp}
            onMouseOut={handleMouseOutCanvas}
            onMouseLeave={handleMouseOutCanvas}
            style={{ cursor: 'none' }}
          >
            <Layer>
              {circles.map((circle, i) => (
                <Circle
                  key={i}
                  x={circle.x}
                  y={circle.y}
                  radius={circle.radius}
                  fill={'rgba(255,255,255,1'}
                  globalCompositeOperation={
                    circle.tool === 'eraser' ? 'destination-out' : 'source-over'
                  }
                />
              ))}
              {lines.map((line, i) => (
                <Line
                  key={i}
                  points={line.points}
                  stroke={'rgba(255,255,255,1)'}
                  strokeWidth={line.strokeWidth * 2}
                  tension={0}
                  lineCap="round"
                  lineJoin="round"
                  globalCompositeOperation={
                    line.tool === 'eraser' ? 'destination-out' : 'source-over'
                  }
                />
              ))}
              {(cursorPos || shouldShowBrushPreview) && (
                <Circle
                  x={cursorPos ? cursorPos.x : canvasBgImage.width / 2}
                  y={cursorPos ? cursorPos.y : canvasBgImage.height / 2}
                  radius={brushRadius}
                  fill={'rgba(255,255,255,1)'}
                  globalCompositeOperation={
                    tool === 'eraser' ? 'destination-out' : 'source-over'
                  }
                />
              )}
              <KonvaImage
                image={canvasBgImage}
                globalCompositeOperation={
                  shouldInvertMask ? 'source-in' : 'source-out'
                }
              />
              {(cursorPos || shouldShowBrushPreview) && (
                <Circle
                  x={cursorPos ? cursorPos.x : canvasBgImage.width / 2}
                  y={cursorPos ? cursorPos.y : canvasBgImage.height / 2}
                  radius={brushRadius}
                  stroke={'rgba(0,0,0,1)'}
                  strokeWidth={1}
                  strokeEnabled={true}
                />
              )}
            </Layer>
          </Stage>
        </div>
      </>
    )
  );
};

export default Inpainting;
