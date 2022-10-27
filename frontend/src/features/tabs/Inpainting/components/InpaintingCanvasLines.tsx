import { Line } from 'react-konva';
import { useAppSelector } from '../../../../app/store';
import { inpaintingCanvasLinesSelector } from '../inpaintingSliceSelectors';

/**
 * Draws the lines which comprise the mask.
 *
 * Uses globalCompositeOperation to handle the brush and eraser tools.
 */
const InpaintingCanvasLines = () => {
  const { lines, maskColorString } = useAppSelector(
    inpaintingCanvasLinesSelector
  );

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
