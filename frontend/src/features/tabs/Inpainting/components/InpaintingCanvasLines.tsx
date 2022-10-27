import { Line } from 'react-konva';
import { RootState, useAppSelector } from '../../../../app/store';

/**
 * Draws the lines which comprise the mask.
 *
 * Uses globalCompositeOperation to handle the brush and eraser tools.
 */
const InpaintingCanvasLines = () => {
  const { lines, maskColor } = useAppSelector(
    (state: RootState) => state.inpainting
  );
  const { r, g, b } = maskColor;
  const maskColorString = `rgb(${r},${g},${b})`;

  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          stroke={maskColorString}
          strokeWidth={line.strokeWidth * 2}
          tension={0}
          lineCap="round"
          lineJoin="round"
          shadowForStrokeEnabled={false}
          listening={false}
          globalCompositeOperation={
            line.tool === 'brush' ? 'source-over' : 'destination-out'
          }
        />
      ))}
    </>
  );
};

export default InpaintingCanvasLines;
