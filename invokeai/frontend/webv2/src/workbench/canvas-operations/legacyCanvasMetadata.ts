import type { AccountScope } from '@platform/state/accountLifecycle';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/api';

import { galleryImageUrls } from '@features/gallery/utility';
import { createItemId } from '@platform/browser/randomUuid';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { apiFetchRaw } from '@platform/transport/http';
import { uploadCanvasImage } from '@workbench/canvas-operations/backend/canvasImages';
import { createEmptyCanvasDocument, isRecord } from '@workbench/canvasMigration';
import { isStorableControlWeight, isStorableControlStepRange } from '@workbench/controlAdapters';
import { getStroke } from 'perfect-freehand';
import { z } from 'zod';

import { applyLegacyCanvasAdjustments } from './legacyCanvasAdjustments';
import { drawLegacyOpacityStroke, prepareLegacyOpacityStroke } from './legacyCanvasPressure';

const number = z.number().finite();
const point = z.object({ x: number, y: number });
const rect = point.extend({ width: number.positive(), height: number.positive() });
const color = z.object({
  r: number.min(0).max(255),
  g: number.min(0).max(255),
  b: number.min(0).max(255),
  a: number.min(0).max(1).default(1),
});
const image = z.object({ image_name: z.string().min(1), width: number.positive(), height: number.positive() });
const objectImage = z.union([
  image,
  z.object({
    dataURL: z.string().regex(/^data:image\/(?:png|jpeg|webp);base64,/u),
    width: number.positive(),
    height: number.positive(),
  }),
]);
const operation = z
  .enum([
    'source-over',
    'source-in',
    'source-out',
    'source-atop',
    'destination-over',
    'destination-in',
    'destination-out',
    'destination-atop',
    'lighter',
    'copy',
    'xor',
    'multiply',
    'screen',
    'overlay',
    'darken',
    'lighten',
    'color-dodge',
    'color-burn',
    'hard-light',
    'soft-light',
    'difference',
    'exclusion',
    'hue',
    'saturation',
    'color',
    'luminosity',
  ])
  .default('source-over');
const common = { id: z.string(), clip: rect.nullable().optional() };
const stroke = {
  ...common,
  points: z.array(number).max(1_000_000),
  strokeWidth: number.positive(),
  pressureAffectsWidth: z.boolean().default(true),
  pressureAffectsOpacity: z.boolean().default(false),
};
const object = z.discriminatedUnion('type', [
  z.object({ ...common, type: z.literal('image'), image: objectImage }),
  z.object({ ...stroke, type: z.literal('brush_line'), color, globalCompositeOperation: operation }),
  z.object({ ...stroke, type: z.literal('eraser_line') }),
  z.object({ ...stroke, type: z.literal('brush_line_with_pressure'), color, globalCompositeOperation: operation }),
  z.object({ ...stroke, type: z.literal('eraser_line_with_pressure') }),
  z.object({ ...common, type: z.literal('rect'), rect, color, compositeOperation: operation }),
  z.object({ ...common, type: z.literal('oval'), rect, color, compositeOperation: operation }),
  z.object({
    ...common,
    type: z.literal('polygon'),
    points: z.array(number).max(1_000_000),
    color,
    compositeOperation: operation,
  }),
  z.object({
    ...common,
    type: z.literal('lasso'),
    points: z.array(number).max(1_000_000),
    compositeOperation: operation,
  }),
  z.object({
    ...common,
    type: z.literal('gradient'),
    bboxRect: rect,
    gradientType: z.enum(['linear', 'radial']),
    rect,
    start: point.optional(),
    end: point.optional(),
    center: point.optional(),
    radius: number.nonnegative().optional(),
    clipCenter: point,
    clipRadius: number.nonnegative(),
    clipAngle: number.optional(),
    clipEnabled: z.boolean().default(true),
    fgColor: color,
    bgColor: color,
    globalCompositeOperation: operation,
  }),
]);
type LegacyObject = z.infer<typeof object>;
const rgba = (c: z.infer<typeof color>): string => `rgba(${c.r},${c.g},${c.b},${c.a})`;
const fill = z.object({ style: z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']), color });
const adapter = z.object({
  type: z.enum(['controlnet', 't2i_adapter', 'control_lora', 'z_image_control', 'anima_lllite']),
  model: z.record(z.string(), z.unknown()).nullable(),
  weight: number.min(-1).max(2),
  beginEndStepPct: z.tuple([number.min(0).max(1), number.min(0).max(1)]).default([0, 1]),
  controlMode: z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']).nullable().default(null),
});
const layer = z.object({
  id: z.string(),
  name: z.string().nullable(),
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
  position: point,
  opacity: number.min(0).max(1),
  objects: z.array(object).max(10000),
  controlAdapter: adapter.optional(),
  withTransparencyEffect: z.boolean().optional(),
  fill: fill.optional(),
  positivePrompt: z.string().nullable().optional(),
  negativePrompt: z.string().nullable().optional(),
  autoNegative: z.boolean().optional(),
  referenceImages: z.array(z.object({ id: z.string(), config: z.record(z.string(), z.unknown()) })).optional(),
  noiseLevel: number.min(0).max(1).optional(),
  denoiseLimit: number.min(0).max(1).optional(),
  globalCompositeOperation: z.string().optional(),
  isTransparencyLocked: z.boolean().optional(),
  adjustments: z.record(z.string(), z.unknown()).optional(),
});

/** Konva's open Line with tension 0.3, used by legacy non-pressure brush and eraser objects. */
const legacyLineControls = (points: number[]) => {
  const controls: { x: number; y: number; beforeX: number; beforeY: number; afterX: number; afterY: number }[] = [];
  for (let i = 2; i < points.length - 2; i += 2) {
    const x0 = points[i - 2]!,
      y0 = points[i - 1]!,
      x = points[i]!,
      y = points[i + 1]!,
      x2 = points[i + 2]!,
      y2 = points[i + 3]!;
    const before = Math.hypot(x - x0, y - y0),
      after = Math.hypot(x2 - x, y2 - y),
      total = before + after;
    const a = total ? (0.3 * before) / total : 0,
      b = total ? (0.3 * after) / total : 0;
    controls.push({
      x,
      y,
      beforeX: x - a * (x2 - x0),
      beforeY: y - a * (y2 - y0),
      afterX: x + b * (x2 - x0),
      afterY: y + b * (y2 - y0),
    });
  }
  return controls;
};

const strokeLegacyLine = (ctx: OffscreenCanvasRenderingContext2D, points: number[]): void => {
  if (points.length < 2) {
    return;
  }
  ctx.moveTo(points[0]!, points[1]!);
  const controls = legacyLineControls(points);
  const first = controls[0];
  if (first) {
    ctx.quadraticCurveTo(first.beforeX, first.beforeY, first.x, first.y);
    for (let i = 1; i < controls.length; i++) {
      const previous = controls[i - 1]!,
        current = controls[i]!;
      ctx.bezierCurveTo(previous.afterX, previous.afterY, current.beforeX, current.beforeY, current.x, current.y);
    }
    const last = controls.at(-1)!;
    ctx.quadraticCurveTo(last.afterX, last.afterY, points.at(-2)!, points.at(-1)!);
  } else if (points.length >= 4) {
    ctx.lineTo(points[2]!, points[3]!);
  }
};

/** Legacy object lists become one durable paint bitmap per layer; unsupported/damaged layers are never replaced by blanks. */
const rasterizeLegacyObjects = async (
  objects: LegacyObject[],
  owner: AccountScope,
  adjustments: unknown,
  isCurrent: () => boolean,
  budget: { pixels: number; uploads: number; workUnits: number },
  createdImageNames: string[]
): Promise<CanvasLayerSourceContract> => {
  if (objects.length === 0) {
    return { type: 'paint', bitmap: null };
  }
  if (
    adjustments === undefined &&
    objects.length === 1 &&
    objects[0]?.type === 'image' &&
    'image_name' in objects[0].image
  ) {
    const ref = objects[0].image;
    return { type: 'image', image: { imageName: ref.image_name, width: ref.width, height: ref.height } };
  }
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  const include = (x: number, y: number, padding = 0): void => {
    minX = Math.min(minX, x - padding);
    minY = Math.min(minY, y - padding);
    maxX = Math.max(maxX, x + padding);
    maxY = Math.max(maxY, y + padding);
  };
  for (const item of objects) {
    if (item.type === 'image') {
      include(0, 0);
      include(item.image.width, item.image.height);
    } else if ('rect' in item) {
      include(item.rect.x, item.rect.y);
      include(item.rect.x + item.rect.width, item.rect.y + item.rect.height);
    } else {
      const stride = item.type.endsWith('_with_pressure') ? 3 : 2;
      if (item.points.length % stride !== 0) {
        throw new Error('invalid-points');
      }
      for (let i = 0; i < item.points.length; i += stride) {
        include(item.points[i]!, item.points[i + 1]!, 'strokeWidth' in item ? item.strokeWidth : 0);
      }
      if (stride === 2 && 'strokeWidth' in item) {
        for (const handle of legacyLineControls(item.points)) {
          include(handle.beforeX, handle.beforeY, item.strokeWidth);
          include(handle.afterX, handle.afterY, item.strokeWidth);
        }
      }
    }
  }
  minX = Math.floor(minX);
  minY = Math.floor(minY);
  const width = Math.ceil(maxX) - minX,
    height = Math.ceil(maxY) - minY;
  // Bound each layer surface; pressure scratch and full-output work are reserved separately below.
  if (
    !Number.isFinite(width * height) ||
    width < 1 ||
    height < 1 ||
    width > 16384 ||
    height > 16384 ||
    width * height > 16_777_216
  ) {
    throw new Error('legacy-layer-too-large');
  }
  if (budget.pixels < width * height || budget.uploads === 0) {
    throw new Error('legacy-conversion-limit');
  }
  assertAccountScopeCurrent(owner);
  if (!isCurrent()) {
    throw new Error('canvas-changed');
  }
  if (budget.workUnits < width * height) {
    throw new Error('legacy-conversion-work-limit');
  }
  budget.workUnits -= width * height;
  const opacityStrokes = new Map<LegacyObject, ReturnType<typeof prepareLegacyOpacityStroke>>();
  // Preparation and reservations stay charged even if this layer is refused later.
  for (const item of objects) {
    if (item.type === 'brush_line_with_pressure' && item.pressureAffectsOpacity) {
      opacityStrokes.set(item, prepareLegacyOpacityStroke(item, { width, height }, budget));
    }
  }
  budget.pixels -= width * height;
  budget.uploads -= 1;
  const canvas = new OffscreenCanvas(width, height);
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('canvas-unavailable');
    }
    ctx.translate(-minX, -minY);
    for (const item of objects) {
      assertAccountScopeCurrent(owner);
      if (!isCurrent()) {
        throw new Error('canvas-changed');
      }
      ctx.save();
      try {
        if (item.clip) {
          ctx.beginPath();
          ctx.rect(item.clip.x, item.clip.y, item.clip.width, item.clip.height);
          ctx.clip();
        }
        ctx.fillStyle = 'color' in item ? rgba(item.color) : 'white';
        ctx.strokeStyle = ctx.fillStyle;
        if ('compositeOperation' in item) {
          ctx.globalCompositeOperation = item.compositeOperation;
        }
        if ('globalCompositeOperation' in item) {
          ctx.globalCompositeOperation = item.globalCompositeOperation;
        }
        if (item.type === 'image') {
          let blob: Blob;
          if ('image_name' in item.image) {
            const response = await apiFetchRaw(galleryImageUrls.full(item.image.image_name), { signal: owner.signal });
            if (!response.ok) {
              throw new Error('image-unavailable');
            }
            blob = await response.blob();
          } else {
            const [prefix, encoded] = item.image.dataURL.split(',');
            const bytes = Uint8Array.from(atob(encoded!), (character) => character.charCodeAt(0));
            blob = new Blob([bytes], { type: prefix!.slice(5, prefix!.indexOf(';')) });
          }
          const bitmap = await createImageBitmap(blob, {
            resizeWidth: Math.round(item.image.width),
            resizeHeight: Math.round(item.image.height),
          });
          try {
            assertAccountScopeCurrent(owner);
            ctx.drawImage(bitmap, 0, 0, item.image.width, item.image.height);
          } finally {
            bitmap.close();
          }
        } else if (item.type === 'rect' || item.type === 'oval') {
          const r = item.rect;
          ctx.beginPath();
          if (item.type === 'rect') {
            ctx.rect(r.x, r.y, r.width, r.height);
          } else {
            ctx.ellipse(r.x + r.width / 2, r.y + r.height / 2, r.width / 2, r.height / 2, 0, 0, 2 * Math.PI);
          }
          ctx.fill();
        } else if (item.type === 'gradient') {
          let gradient: CanvasGradient;
          if (item.gradientType === 'linear' && item.start && item.end) {
            gradient = ctx.createLinearGradient(
              item.rect.x + item.start.x,
              item.rect.y + item.start.y,
              item.rect.x + item.end.x,
              item.rect.y + item.end.y
            );
          } else if (item.center && item.radius) {
            gradient = ctx.createRadialGradient(
              item.rect.x + item.center.x,
              item.rect.y + item.center.y,
              0,
              item.rect.x + item.center.x,
              item.rect.y + item.center.y,
              item.radius
            );
          } else {
            throw new Error('invalid-gradient');
          }
          gradient.addColorStop(0, rgba(item.fgColor));
          gradient.addColorStop(1, rgba(item.bgColor));
          ctx.beginPath();
          ctx.rect(item.bboxRect.x, item.bboxRect.y, item.bboxRect.width, item.bboxRect.height);
          ctx.clip();
          if (item.clipEnabled) {
            ctx.beginPath();
            if (item.gradientType === 'radial') {
              ctx.arc(item.clipCenter.x, item.clipCenter.y, item.clipRadius, 0, 2 * Math.PI);
            } else if (item.start && item.end) {
              const dx = Math.cos(item.clipAngle ?? 0),
                dy = Math.sin(item.clipAngle ?? 0);
              const extent = Math.max(Math.hypot(item.bboxRect.width, item.bboxRect.height) * 2, 1);
              const halfPlane = (x: number, y: number, sign: number): void => {
                ctx.moveTo(x - dy * extent, y + dx * extent);
                ctx.lineTo(x + dy * extent, y - dx * extent);
                ctx.lineTo(x + dy * extent + dx * sign * extent, y - dx * extent + dy * sign * extent);
                ctx.lineTo(x - dy * extent + dx * sign * extent, y + dx * extent + dy * sign * extent);
                ctx.closePath();
              };
              halfPlane(item.rect.x + item.start.x, item.rect.y + item.start.y, 1);
              ctx.clip();
              ctx.beginPath();
              halfPlane(item.rect.x + item.end.x, item.rect.y + item.end.y, -1);
            }
            ctx.clip();
          }
          ctx.fillStyle = gradient;
          ctx.fillRect(item.rect.x, item.rect.y, item.rect.width, item.rect.height);
        } else if (item.type === 'polygon' || item.type === 'lasso') {
          if (item.points.length % 2 !== 0) {
            throw new Error('invalid-points');
          }
          ctx.beginPath();
          for (let i = 0; i < item.points.length; i += 2) {
            if (i === 0) {
              ctx.moveTo(item.points[i]!, item.points[i + 1]!);
            } else {
              ctx.lineTo(item.points[i]!, item.points[i + 1]!);
            }
          }
          ctx.closePath();
          ctx.fill();
        } else {
          if (item.type.startsWith('eraser')) {
            ctx.globalCompositeOperation = 'destination-out';
          }
          if (item.type === 'brush_line_with_pressure' && item.pressureAffectsOpacity) {
            drawLegacyOpacityStroke(ctx, opacityStrokes.get(item)!, { x: minX, y: minY, width, height });
          } else if (item.type.endsWith('_with_pressure')) {
            const points: number[][] = [];
            for (let i = 0; i < item.points.length; i += 3) {
              points.push([item.points[i]!, item.points[i + 1]!, Math.max(0, Math.min(1, item.points[i + 2]!))]);
            }
            const outline = getStroke(points, {
              size: item.strokeWidth / 2,
              thinning: item.pressureAffectsWidth ? 1 : 0,
              simulatePressure: false,
              last: true,
            });
            if (outline.length >= 4) {
              const [first, second, third] = outline;
              ctx.beginPath();
              ctx.moveTo(first![0]!, first![1]!);
              ctx.quadraticCurveTo(
                second![0]!,
                second![1]!,
                (second![0]! + third![0]!) / 2,
                (second![1]! + third![1]!) / 2
              );
              for (let i = 2; i < outline.length - 1; i++) {
                const a = outline[i]!,
                  b = outline[i + 1]!;
                ctx.quadraticCurveTo(a[0]!, a[1]!, (a[0]! + b[0]!) / 2, (a[1]! + b[1]!) / 2);
              }
              ctx.closePath();
              ctx.fill();
            }
          } else {
            ctx.lineWidth = item.strokeWidth;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            strokeLegacyLine(ctx, item.points);
            if (item.points.length === 2) {
              ctx.arc(item.points[0]!, item.points[1]!, item.strokeWidth / 2, 0, 2 * Math.PI);
              ctx.fill();
            } else {
              ctx.stroke();
            }
          }
        }
      } finally {
        ctx.restore();
      }
    }
    if (adjustments !== undefined) {
      const pixels = ctx.getImageData(0, 0, width, height);
      applyLegacyCanvasAdjustments(pixels, adjustments);
      ctx.putImageData(pixels, 0, 0);
    }
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    assertAccountScopeCurrent(owner);
    if (!isCurrent()) {
      throw new Error('canvas-changed');
    }
    // Until the recall commits, even an upload whose response is lost must stay reclaimable.
    const uploaded = await uploadCanvasImage(blob, { isIntermediate: true, signal: owner.signal });
    createdImageNames.push(uploaded.imageName);
    assertAccountScopeCurrent(owner);
    if (!isCurrent()) {
      throw new Error('canvas-changed');
    }
    return { type: 'paint', bitmap: uploaded, offset: { x: minX, y: minY } };
  } finally {
    canvas.width = 1;
    canvas.height = 1;
  }
};

export const convertLegacyCanvasMetadata = async (
  metadata: Record<string, unknown>,
  owner: AccountScope,
  isCurrent: () => boolean = () => true
): Promise<{ record: Record<string, unknown> | null; problems: string[]; createdImageNames: string[] }> => {
  const raw = metadata.canvas_v2_metadata;
  const budget = { pixels: 64 * 1024 * 1024, uploads: 128, workUnits: 64 * 1024 * 1024 };
  const problems: string[] = [];
  const createdImageNames: string[] = [];
  const document = createEmptyCanvasDocument(
    typeof metadata.width === 'number' ? metadata.width : 1024,
    typeof metadata.height === 'number' ? metadata.height : 1024
  );
  const controlModels: Record<string, unknown> = Object.create(null);
  if (!isRecord(raw)) {
    // recall_parameters names already prepared control maps, unlike legacy authored object lists.
    if (!Array.isArray(metadata.control_layers)) {
      return { record: null, problems: 'control_layers' in metadata ? ['invalid-control'] : [], createdImageNames };
    }
    for (const control of metadata.control_layers.slice(0, 10000)) {
      if (
        !isRecord(control) ||
        !isRecord(control.image) ||
        !(typeof control.model_key === 'string' || isRecord(control.model))
      ) {
        problems.push('invalid-control');
        continue;
      }
      const weight = control.weight === undefined ? 1 : control.weight;
      const range = [
        control.begin_step_percent === undefined ? 0 : control.begin_step_percent,
        control.end_step_percent === undefined ? 1 : control.end_step_percent,
      ];
      if (!isStorableControlWeight(weight) || !isStorableControlStepRange(range)) {
        problems.push('invalid-control');
        continue;
      }
      const asset = image.safeParse(control.processed_image ?? control.image);
      if (!asset.success) {
        problems.push('invalid-control-image');
        continue;
      }
      const id = createItemId('control');
      const modelKey =
        typeof control.model_key === 'string'
          ? control.model_key
          : isRecord(control.model) && typeof control.model.key === 'string'
            ? control.model.key
            : null;
      if (isRecord(control.model)) {
        controlModels[id] = control.model;
      }
      document.stacks.control.push({
        id,
        name: typeof control.name === 'string' ? control.name : 'Control layer',
        isEnabled: control.is_enabled !== false,
        isLocked: false,
        opacity: 1,
        blendMode: 'normal',
        transform: {
          x: isRecord(control.position) && typeof control.position.x === 'number' ? control.position.x : 0,
          y: isRecord(control.position) && typeof control.position.y === 'number' ? control.position.y : 0,
          rotation: 0,
          scaleX: 1,
          scaleY: 1,
        },
        type: 'control',
        source: {
          type: 'image',
          image: { imageName: asset.data.image_name, width: asset.data.width, height: asset.data.height },
        },
        withTransparencyEffect: false,
        adapter: {
          model: modelKey,
          kind: 'controlnet',
          weight,
          beginEndStepPct: range,
          controlMode:
            control.control_mode === 'more_prompt' ||
            control.control_mode === 'more_control' ||
            control.control_mode === 'unbalanced'
              ? control.control_mode
              : 'balanced',
        },
      });
    }
    if (metadata.control_layers.length && !document.stacks.control.length) {
      return { record: null, problems, createdImageNames };
    }
  } else {
    const stacks = {
      rasterLayers: 'raster',
      controlLayers: 'control',
      inpaintMasks: 'inpaint_mask',
      regionalGuidance: 'regional_guidance',
    } as const;
    if (
      !Object.keys(stacks).every((key) => Array.isArray(raw[key])) ||
      Object.keys(stacks).reduce((sum, key) => sum + (raw[key] as unknown[]).length, 0) > 10000
    ) {
      return { record: null, problems: ['invalid-legacy-document'], createdImageNames };
    }
    for (const [key, stack] of Object.entries(stacks)) {
      // Legacy arrays are bottom-first; current forests are top-first.
      for (const value of [...(raw[key] as unknown[])].reverse()) {
        if (!isCurrent()) {
          throw new Error('canvas-changed');
        }
        const parsed = layer.safeParse(value);
        if (!parsed.success) {
          problems.push(isRecord(value) && typeof value.name === 'string' ? value.name : 'invalid-legacy-layer');
          continue;
        }
        const item = parsed.data;
        try {
          // Unsupported composite/adjustment semantics must never silently produce a different image.
          if (
            item.globalCompositeOperation &&
            ![
              'source-over',
              'multiply',
              'screen',
              'overlay',
              'darken',
              'lighten',
              'color-dodge',
              'color-burn',
              'hard-light',
              'soft-light',
              'difference',
              'exclusion',
              'hue',
              'saturation',
              'color',
              'luminosity',
            ].includes(item.globalCompositeOperation)
          ) {
            throw new Error('unsupported-blend');
          }
          const source = await rasterizeLegacyObjects(
            item.objects,
            owner,
            item.adjustments,
            isCurrent,
            budget,
            createdImageNames
          );
          assertAccountScopeCurrent(owner);
          const base = {
            id: item.id,
            name: item.name ?? 'Recalled layer',
            isEnabled: item.isEnabled,
            isLocked: item.isLocked,
            opacity: item.opacity,
            blendMode:
              item.globalCompositeOperation === 'source-over' || !item.globalCompositeOperation
                ? 'normal'
                : item.globalCompositeOperation,
            transform: { x: item.position.x, y: item.position.y, scaleX: 1, scaleY: 1, rotation: 0 },
          };
          let result: unknown;
          if (stack === 'raster') {
            result = { ...base, type: stack, source, isTransparencyLocked: item.isTransparencyLocked };
          } else if (stack === 'control' && item.controlAdapter) {
            controlModels[item.id] = item.controlAdapter.model;
            result = {
              ...base,
              type: stack,
              source,
              withTransparencyEffect: item.withTransparencyEffect ?? false,
              adapter: {
                ...item.controlAdapter,
                kind: item.controlAdapter.type,
                model: item.controlAdapter.model?.key ?? null,
              },
            };
          } else {
            if (source.type !== 'image' && source.type !== 'paint') {
              throw new Error('invalid-mask');
            }
            const mask = {
              bitmap: source.type === 'image' ? source.image : source.bitmap,
              offset: source.type === 'paint' ? source.offset : undefined,
              fill: { style: item.fill?.style ?? 'solid', color: item.fill ? rgba(item.fill.color) : 'red' },
            };
            if (stack === 'inpaint_mask') {
              result = {
                ...base,
                type: stack,
                mask,
                ...(item.noiseLevel === undefined ? {} : { noise: { level: item.noiseLevel, isEnabled: true } }),
                ...(item.denoiseLimit === undefined ? {} : { denoise: { limit: item.denoiseLimit, isEnabled: true } }),
              };
            } else {
              result = {
                ...base,
                type: stack,
                mask,
                positivePrompt: item.positivePrompt ?? null,
                negativePrompt: item.negativePrompt ?? null,
                autoNegative: item.autoNegative ?? false,
                referenceImages: (item.referenceImages ?? []).map((ref) => {
                  const asset = image.safeParse(ref.config.image);
                  return {
                    ...ref,
                    isEnabled: true,
                    config: {
                      ...ref.config,
                      image: asset.success
                        ? {
                            imageName: asset.data.image_name,
                            width: asset.data.width,
                            height: asset.data.height,
                            imageUrl: '',
                            thumbnailUrl: '',
                            queuedAt: '',
                            sourceQueueItemId: '',
                          }
                        : null,
                    },
                  };
                }),
              };
            }
          }
          // The current canvas loader below validates this wire projection before anything can commit.
          document.stacks[stack as keyof CanvasDocumentContractV3['stacks']].push(result as CanvasLayerContract);
        } catch (error) {
          assertAccountScopeCurrent(owner);
          if (!isCurrent()) {
            throw new Error('canvas-changed');
          }
          problems.push(`${item.name ?? item.id}: ${error instanceof Error ? error.message : 'unavailable'}`);
        }
      }
    }
    if (Object.values(document.stacks).every((nodes) => nodes.length === 0)) {
      return { record: null, problems, createdImageNames };
    }
  }
  return { record: { version: 1, document, controlModels }, problems, createdImageNames };
};
