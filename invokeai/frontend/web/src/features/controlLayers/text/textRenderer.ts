import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { Coordinate, RgbaColor } from 'features/controlLayers/store/types';
import type { TextAlignment } from 'features/controlLayers/text/textConstants';

type TextRenderConfig = {
  text: string;
  fontSize: number;
  fontFamily: string;
  fontWeight: number;
  fontStyle: 'normal' | 'italic';
  underline: boolean;
  strikethrough: boolean;
  lineHeight: number;
  color: RgbaColor;
  alignment: TextAlignment;
  padding: number;
  devicePixelRatio: number;
};

export type TextMeasureConfig = {
  text: string;
  fontSize: number;
  fontFamily: string;
  fontWeight: number;
  fontStyle: 'normal' | 'italic';
  lineHeight: number;
};

type TextMetrics = {
  lines: string[];
  lineWidths: number[];
  lineHeightPx: number;
  contentWidth: number;
  contentHeight: number;
  ascent: number;
  descent: number;
  baselineOffset: number;
};

type TextRenderResult = {
  canvas: HTMLCanvasElement;
  contentWidth: number;
  contentHeight: number;
  totalWidth: number;
  totalHeight: number;
};

export const renderTextToCanvas = (config: TextRenderConfig): TextRenderResult => {
  const measurement = measureTextContent(config);
  const totalWidth = Math.ceil(measurement.contentWidth + config.padding * 2);
  const totalHeight = Math.ceil(measurement.contentHeight + config.padding * 2);

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Unable to acquire 2D context');
  }
  const dpr = Math.max(1, config.devicePixelRatio);
  canvas.width = Math.max(1, Math.ceil(totalWidth * dpr));
  canvas.height = Math.max(1, Math.ceil(totalHeight * dpr));
  canvas.style.width = `${totalWidth}px`;
  canvas.style.height = `${totalHeight}px`;
  ctx.scale(dpr, dpr);
  ctx.font = buildFontDescriptor(config);
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = rgbaColorToString(config.color);
  const dprScale = Math.max(1, config.devicePixelRatio);

  measurement.lines.forEach((line, index) => {
    const text = line === '' ? ' ' : line;
    const lineWidth = measurement.lineWidths[index] ?? 0;
    const x = computeAlignedX(lineWidth, measurement.contentWidth, config.alignment, config.padding);
    const y = config.padding + measurement.baselineOffset + index * measurement.lineHeightPx;
    const snappedX = snapToDpr(x, dprScale);
    const snappedY = snapToDpr(y, dprScale);
    ctx.fillText(text, snappedX, snappedY);
    if (config.underline) {
      const underlineY = snapToDpr(snappedY + config.fontSize + 2, dprScale);
      ctx.fillRect(snappedX, underlineY, lineWidth, Math.max(1, config.fontSize * 0.08));
    }
    if (config.strikethrough) {
      const strikeY = snapToDpr(snappedY + config.fontSize * 0.55, dprScale);
      ctx.fillRect(snappedX, strikeY, lineWidth, Math.max(1, config.fontSize * 0.08));
    }
  });

  return {
    canvas,
    contentWidth: measurement.contentWidth,
    contentHeight: measurement.contentHeight,
    totalWidth,
    totalHeight,
  };
};

export const measureTextContent = (config: TextMeasureConfig): TextMetrics => {
  const lines = config.text.split(/\r?\n/);
  const fontDescriptor = buildFontDescriptor(config);
  const measurementCanvas = document.createElement('canvas');
  const measureCtx = measurementCanvas.getContext('2d');
  if (!measureCtx) {
    throw new Error('Failed to build 2D context');
  }
  measureCtx.font = fontDescriptor;
  const sampleMetrics = measureCtx.measureText('Mg');
  const fallbackAscent = config.fontSize * 0.8;
  const fallbackDescent = config.fontSize * 0.2;
  const ascent = sampleMetrics.actualBoundingBoxAscent || fallbackAscent;
  const descent = sampleMetrics.actualBoundingBoxDescent || fallbackDescent;
  const lineHeightPx = (ascent + descent) * config.lineHeight;
  const extraLeading = Math.max(0, lineHeightPx - (ascent + descent));
  const baselineOffset = ascent + extraLeading / 2;
  const lineWidths = lines.map((line) => measureCtx.measureText(line === '' ? ' ' : line).width);
  const contentWidth = Math.max(...lineWidths, config.fontSize);
  const contentHeight = Math.max(lines.length, 1) * lineHeightPx;
  return {
    lines,
    lineWidths,
    lineHeightPx,
    contentWidth,
    contentHeight,
    ascent,
    descent,
    baselineOffset,
  };
};

export const computeAlignedX = (lineWidth: number, contentWidth: number, alignment: TextAlignment, padding: number) => {
  if (alignment === 'center') {
    return padding + (contentWidth - lineWidth) / 2;
  }
  if (alignment === 'right') {
    return padding + (contentWidth - lineWidth);
  }
  return padding;
};

export const buildFontDescriptor = (config: {
  fontStyle: 'normal' | 'italic';
  fontWeight: number;
  fontSize: number;
  fontFamily: string;
}) => {
  const weight = config.fontWeight || 400;
  return `${config.fontStyle === 'italic' ? 'italic ' : ''}${weight} ${config.fontSize}px ${config.fontFamily}`;
};

export const calculateLayerPosition = (
  anchor: Coordinate,
  alignment: TextAlignment,
  contentWidth: number,
  padding: number
) => {
  let offsetX = -padding;
  if (alignment === 'center') {
    offsetX = -(contentWidth / 2) - padding;
  } else if (alignment === 'right') {
    offsetX = -contentWidth - padding;
  }
  return {
    x: anchor.x + offsetX,
    y: anchor.y - padding,
  };
};

export const hasVisibleGlyphs = (text: string): boolean => {
  return text.replace(/\s+/g, '').length > 0;
};

const snapToDpr = (value: number, dpr: number): number => {
  return Math.round(value * dpr) / dpr;
};
