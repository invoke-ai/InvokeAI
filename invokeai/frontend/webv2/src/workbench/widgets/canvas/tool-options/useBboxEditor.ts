import {
  bboxEquals,
  roundBbox,
  type CanvasCoreStoreCapability,
  type CanvasLayerCapability,
  type Rect,
} from '@workbench/canvas-engine/api';
import { useBboxGrid, useBboxOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export type BboxEditorEngine = CanvasCoreStoreCapability & { readonly layers: CanvasLayerCapability };

const bboxEqualsSelected = (a: Rect | null, b: Rect | null): boolean =>
  a === b || (!!a && !!b && a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height);

export const useBboxEditor = (engine: BboxEditorEngine) => {
  const { t } = useTranslation();
  const bbox = useActiveProjectSelector((project) => project.canvas.document.bbox, bboxEqualsSelected);
  const options = useBboxOptions(engine);
  const grid = useBboxGrid(engine);

  const commitBbox = useCallback(
    (next: Rect) => {
      const rounded = roundBbox(next);
      if (bboxEquals(rounded, bbox)) {
        return;
      }
      engine.layers.commitStructural(
        t('widgets.canvas.toolOptions.setFrame'),
        { bbox: rounded, type: 'setCanvasBbox' },
        { bbox: roundBbox(bbox), type: 'setCanvasBbox' }
      );
    },
    [bbox, engine, t]
  );

  const setWidth = useCallback(
    (value: number) => {
      const width = Math.max(1, Math.round(value / grid) * grid);
      const height =
        options.aspectLocked && options.aspectRatio > 0
          ? Math.max(1, Math.round(width / options.aspectRatio))
          : bbox.height;
      commitBbox({ height, width, x: bbox.x, y: bbox.y });
    },
    [bbox, commitBbox, grid, options.aspectLocked, options.aspectRatio]
  );

  const setHeight = useCallback(
    (value: number) => {
      const height = Math.max(1, Math.round(value / grid) * grid);
      const width =
        options.aspectLocked && options.aspectRatio > 0
          ? Math.max(1, Math.round(height * options.aspectRatio))
          : bbox.width;
      commitBbox({ height, width, x: bbox.x, y: bbox.y });
    },
    [bbox, commitBbox, grid, options.aspectLocked, options.aspectRatio]
  );

  const setX = useCallback(
    (value: number) => commitBbox({ ...bbox, x: Math.round(value / grid) * grid }),
    [bbox, commitBbox, grid]
  );

  const setY = useCallback(
    (value: number) => commitBbox({ ...bbox, y: Math.round(value / grid) * grid }),
    [bbox, commitBbox, grid]
  );

  return { bbox, commitBbox, grid, options, setHeight, setWidth, setX, setY };
};
