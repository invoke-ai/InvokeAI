// import { GroupConfig } from 'konva/lib/Group';
// import { Group, Line } from 'react-konva';
// import { RootState, useAppSelector } from 'app/store';
// import { createSelector } from '@reduxjs/toolkit';
// import { OutpaintingCanvasState } from './canvasSlice';

// export const canvasEraserLinesSelector = createSelector(
//   (state: RootState) => state.canvas.outpainting,
//   (outpainting: OutpaintingCanvasState) => {
//     const { eraserLines } = outpainting;
//     return {
//       eraserLines,
//     };
//   }
// );

// type IAICanvasEraserLinesProps = GroupConfig;

// /**
//  * Draws the lines which comprise the mask.
//  *
//  * Uses globalCompositeOperation to handle the brush and eraser tools.
//  */
// const IAICanvasEraserLines = (props: IAICanvasEraserLinesProps) => {
//   const { ...rest } = props;
//   const { eraserLines } = useAppSelector(canvasEraserLinesSelector);

//   return (
//     <Group {...rest} globalCompositeOperation={'destination-out'}>
//       {eraserLines.map((line, i) => (
//         <Line
//           key={i}
//           points={line.points}
//           stroke={'rgb(0,0,0)'} // The lines can be any color, just need alpha > 0
//           strokeWidth={line.strokeWidth * 2}
//           tension={0}
//           lineCap="round"
//           lineJoin="round"
//           shadowForStrokeEnabled={false}
//           listening={false}
//           globalCompositeOperation={'source-over'}
//         />
//       ))}
//     </Group>
//   );
// };

// export default IAICanvasEraserLines;
export default {}