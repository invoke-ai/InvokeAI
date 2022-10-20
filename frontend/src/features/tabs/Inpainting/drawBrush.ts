export type BrushShape = 'circle' | 'square';

const drawBrush = (
  context: CanvasRenderingContext2D,
  brush: BrushShape,
  size: number,
  x: number,
  y: number,
  outline?: {
    lineDashOffset: number;
    lineDash: number[];
    strokeStyle: string;
    lineWidth: number;
  }
) => {
  console.log('brush: ', brush)

  switch (brush) {
    case 'circle': {
      context.beginPath();
      context.arc(x, y, size / 2, 0, Math.PI * 2);
      if (outline) {
        const { lineDashOffset, lineDash, strokeStyle, lineWidth } = outline;
        context.lineDashOffset = lineDashOffset;
        context.setLineDash(lineDash);
        context.strokeStyle = strokeStyle;
        context.lineWidth = lineWidth;
        context.stroke();
      } else {
        context.fill();
      }
      context.closePath();
      break;
    }
    case 'square': {
      context.beginPath();
      context.rect(x - size / 2, y - size / 2, size, size);
      if (outline) {
        const { lineDashOffset, lineDash, strokeStyle, lineWidth } = outline;
        context.lineDashOffset = lineDashOffset;
        context.setLineDash(lineDash);
        context.strokeStyle = strokeStyle;
        context.lineWidth = lineWidth;
        context.stroke();
      } else {
        context.fill();
      }
      context.closePath();
    }
  }
};

export default drawBrush;
