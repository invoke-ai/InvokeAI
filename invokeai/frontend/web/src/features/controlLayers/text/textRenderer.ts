import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { Coordinate, RgbaColor } from 'features/controlLayers/store/types';
import type { TextAlignment } from 'features/controlLayers/text/textConstants';

export type TextRenderConfig = {
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

export type TextMetrics = {
  lines: string[];
  lineWidths: number[];
  lineHeightPx: number;
  contentWidth: number;
  contentHeight: number;
};

export type TextRenderResult = {
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
  ctx.textBaseline = 'top';
  ctx.fillStyle = rgbaColorToString(config.color);

  measurement.lines.forEach((line, index) => {
    const text = line === '' ? ' ' : line;
    const lineWidth = measurement.lineWidths[index] ?? 0;
    const x = computeAlignedX(lineWidth, measurement.contentWidth, config.alignment, config.padding);
    const y = config.padding + index * measurement.lineHeightPx;
    ctx.fillText(text, x, y);
    if (config.underline) {
      const underlineY = y + config.fontSize + 2;
      ctx.fillRect(x, underlineY, lineWidth, Math.max(1, config.fontSize * 0.08));
    }
    if (config.strikethrough) {
      const strikeY = y + config.fontSize * 0.55;
      ctx.fillRect(x, strikeY, lineWidth, Math.max(1, config.fontSize * 0.08));
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
  const lineHeightPx = config.fontSize * config.lineHeight;
  const lineWidths = lines.map((line) => measureCtx.measureText(line === '' ? ' ' : line).width);
  const contentWidth = Math.max(...lineWidths, config.fontSize);
  const contentHeight = Math.max(lines.length, 1) * lineHeightPx;
  return {
    lines,
    lineWidths,
    lineHeightPx,
    contentWidth,
    contentHeight,
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

const buildFontDescriptor = (config: { fontStyle: 'normal' | 'italic'; fontWeight: number; fontSize: number; fontFamily: string }) => {
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
    x: Math.round(anchor.x + offsetX),
    y: Math.round(anchor.y - padding),
  };
};

export const hasVisibleGlyphs = (text: string): boolean => {
  return text.replace(/\s+/g, '').length > 0;
};
