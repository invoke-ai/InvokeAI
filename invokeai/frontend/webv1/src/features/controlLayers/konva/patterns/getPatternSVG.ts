import { rgbColorToString } from 'common/util/colorCodeTransformers';
import type { FillStyle, RgbColor } from 'features/controlLayers/store/types';

import crosshatch from './pattern-crosshatch.svg?raw';
import diagonal from './pattern-diagonal.svg?raw';
import grid from './pattern-grid.svg?raw';
import horizontal from './pattern-horizontal.svg?raw';
import vertical from './pattern-vertical.svg?raw';

export function getPatternSVG(pattern: Exclude<FillStyle, 'solid'>, color: RgbColor) {
  let content: string = 'data:image/svg+xml;utf8,';
  if (pattern === 'crosshatch') {
    content += crosshatch;
  } else if (pattern === 'diagonal') {
    content += diagonal;
  } else if (pattern === 'horizontal') {
    content += horizontal;
  } else if (pattern === 'vertical') {
    content += vertical;
  } else if (pattern === 'grid') {
    content += grid;
  }

  content = content.replaceAll('stroke:black', `stroke:${rgbColorToString(color)}`);

  return content;
}
