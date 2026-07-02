import type { ClipVisionModel, IPAdapterMethod } from '@workbench/generation/types';

/** The coarse mode shown in the segmented control. `style` covers all style method variants. */
export type ReferenceMode = 'style' | 'composition' | 'full';

export type StyleMethod = Extract<IPAdapterMethod, 'style' | 'style_strong' | 'style_precise'>;

export const MODE_SEGMENTS: { labelKey: string; value: ReferenceMode }[] = [
  { labelKey: 'widgets.generate.modeStyle', value: 'style' },
  { labelKey: 'widgets.generate.modeComposition', value: 'composition' },
  { labelKey: 'widgets.generate.modeBoth', value: 'full' },
];

export const STYLE_VARIANT_OPTIONS: { descriptionKey: string; labelKey: string; value: StyleMethod }[] = [
  {
    descriptionKey: 'widgets.generate.styleVariantStandardDesc',
    labelKey: 'widgets.generate.styleVariantStandard',
    value: 'style',
  },
  {
    descriptionKey: 'widgets.generate.styleVariantStrongDesc',
    labelKey: 'widgets.generate.styleVariantStrong',
    value: 'style_strong',
  },
  {
    descriptionKey: 'widgets.generate.styleVariantPreciseDesc',
    labelKey: 'widgets.generate.styleVariantPrecise',
    value: 'style_precise',
  },
];

export const CLIP_VISION_MODELS: ClipVisionModel[] = ['ViT-H', 'ViT-G', 'ViT-L'];

export const isReferenceMode = (value: unknown): value is ReferenceMode =>
  value === 'style' || value === 'composition' || value === 'full';

export const isStyleMethod = (method: IPAdapterMethod): method is StyleMethod =>
  method === 'style' || method === 'style_strong' || method === 'style_precise';

export const getReferenceMode = (method: IPAdapterMethod): ReferenceMode => (isStyleMethod(method) ? 'style' : method);

export const getModeLabelKey = (method: IPAdapterMethod): string =>
  MODE_SEGMENTS.find((segment) => segment.value === getReferenceMode(method))?.labelKey ?? '';

export const isClipVisionModel = (value: unknown): value is ClipVisionModel =>
  typeof value === 'string' && CLIP_VISION_MODELS.includes(value as ClipVisionModel);

export const formatWeight = (value: number): string => value.toFixed(2);

export const formatPct = (value: number): string => `${Math.round(value * 100)}%`;
