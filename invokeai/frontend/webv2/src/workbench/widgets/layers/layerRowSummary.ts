import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { getControlImageState } from '@workbench/canvasLayerContent';

export const layerRowSummary = (layer: CanvasLayerContract, t: TFunction): string => {
  switch (layer.type) {
    case 'raster': {
      const parts = [t(`widgets.layers.types.${layer.source.type === 'image' ? 'image' : layer.source.type}`)];
      if (layer.blendMode !== 'normal') {
        parts.push(t(`widgets.layers.blendModes.${layer.blendMode}`));
      }
      if (layer.opacity < 1) {
        parts.push(`${Math.round(layer.opacity * 100)}%`);
      }
      return parts.join(' · ');
    }
    case 'control': {
      if (!layer.controlType) {
        // A legacy layer with its own model works untyped, so it reads as before; a new one asks for its type.
        return layer.adapter.model
          ? `${t(`widgets.layers.control.kinds.${layer.adapter.kind}`)} · ${layer.adapter.weight.toFixed(2)}`
          : t('widgets.layers.control.summary.noType');
      }
      // Judged from the document alone: unsaved strokes read as no image until they persist.
      if (getControlImageState(layer, false) === 'empty') {
        return t('widgets.layers.control.summary.noImage');
      }
      const detector = layer.detection
        ? t(`widgets.layers.control.filters.${layer.detection.type}`, layer.detection.type)
        : t('common.none');
      return [t(`widgets.layers.control.types.${layer.controlType}`), detector, layer.adapter.weight.toFixed(2)].join(
        ' · '
      );
    }
    case 'regional_guidance': {
      const prompt = layer.positivePrompt?.trim();
      if (prompt) {
        return prompt;
      }
      const references = layer.referenceImages.length;
      return references > 0
        ? t('widgets.layers.summary.referenceImages', { count: references })
        : t('widgets.layers.types.regional_guidance');
    }
    case 'inpaint_mask': {
      const parts = [t('widgets.layers.types.inpaint_mask')];
      if (layer.denoise?.isEnabled) {
        parts.push(t('widgets.layers.summary.denoiseLimit', { value: Math.round(layer.denoise.limit * 100) }));
      }
      if (layer.noise?.isEnabled && layer.noise.level > 0) {
        parts.push(t('widgets.layers.summary.noise', { value: Math.round(layer.noise.level * 100) }));
      }
      return parts.join(' · ');
    }
  }
};
