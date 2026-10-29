import type { GenerateComponentValueKey } from './baseGenerationPolicies';
import type { GenerateSettings } from './types';

/**
 * The Generate panel section that owns a setting. Section ids double as the persisted open/closed preference keys, so
 * an issue's target names the section a surface opens to show it.
 */
export type GenerateSectionId =
  | 'advanced'
  | 'components'
  | 'dimensions'
  | 'guidance'
  | 'model'
  | 'providerOptions'
  | 'refiner'
  | 'render';

/**
 * The controls an issue can point at. Each has a renderer that shows the issue there (see `ISSUE_FIELD_RENDERERS` in
 * the Generate UI); a component slot shows its own through the slot picker.
 */
export type GenerationIssueField =
  | GenerateComponentValueKey
  | 'cfgScale'
  | 'concepts'
  | 'krea2RebalanceWeights'
  | 'mossImageUncondSkipLayers'
  /** One reference image's card; the issue's `referenceImageId` names which. */
  | 'referenceImage'
  /** The reference images as a group: their count against the model's limit. */
  | 'referenceImages';

/** Settings an issue can be fixed with in one step; `getGenerationIssueFixPatch` turns one into a settings patch. */
export type GenerationIssueFix =
  | { kind: 'clampGuidance'; value: number }
  | { kind: 'clearModel'; key: GenerationIssueModelSlotKey }
  | { kind: 'removeConcept'; modelKey: string }
  | { aspectRatioValue?: number; height: number; kind: 'setDimensions'; width: number }
  | { kind: 'turnOffPid' }
  | { kind: 'turnOffReferenceImages'; referenceImageIds: readonly string[] };

/** Settings that hold one model reference and can be cleared on their own. */
export type GenerationIssueModelSlotKey = {
  [Key in keyof GenerateSettings]: GenerateSettings[Key] extends { key: string } | null ? Key : never;
}[keyof GenerateSettings];

/**
 * One reason Generate cannot run. `message` is English for logs, thrown compile errors and surfaces without a
 * translator; `messageKey` with `messageValues` is the same text in the locale, when there is one.
 */
export interface GenerationIssue {
  /** Stable for the same problem while it lasts, so surfaces can key, dedupe and jump to it. */
  id: string;
  message: string;
  messageKey?: string;
  messageValues?: Readonly<Record<string, number | string>>;
  /**
   * Where the problem is fixed. `field` names a control that shows the message itself (a component slot, the guidance
   * scrubber), or a group a section lists its issues under; without it the section lists the issue at its top.
   */
  target: { field?: GenerationIssueField; referenceImageId?: string; section: GenerateSectionId };
  fix?: GenerationIssueFix;
}

/** The issue text in the active language; issues without a locale entry fall back to their English. */
export const describeGenerationIssue = (
  t: (key: string, values?: Record<string, unknown>) => string,
  issue: Pick<GenerationIssue, 'message' | 'messageKey' | 'messageValues'>
): string => (issue.messageKey ? t(issue.messageKey, { ...issue.messageValues }) : issue.message);

/** The patch that applies `fix` to `settings`, read when the fix runs so it acts on the latest values. */
export const getGenerationIssueFixPatch = (
  settings: Pick<GenerateSettings, 'loras' | 'referenceImages'>,
  fix: GenerationIssueFix
): Partial<GenerateSettings> => {
  switch (fix.kind) {
    case 'clampGuidance':
      return { cfgScale: fix.value };
    case 'clearModel':
      return { [fix.key]: null };
    case 'removeConcept':
      return { loras: settings.loras.filter((lora) => lora.model.key !== fix.modelKey) };
    case 'setDimensions':
      return {
        height: fix.height,
        width: fix.width,
        ...(fix.aspectRatioValue === undefined ? {} : { aspectRatioValue: fix.aspectRatioValue }),
      };
    case 'turnOffPid':
      return { pidMode: 'off' };
    case 'turnOffReferenceImages': {
      const ids = new Set(fix.referenceImageIds);

      return {
        referenceImages: settings.referenceImages.map((referenceImage) =>
          ids.has(referenceImage.id) ? { ...referenceImage, isEnabled: false } : referenceImage
        ),
      };
    }
  }
};

/** An issue whose text exists only in English: family-specific rules the locale does not cover yet. */
export const createUntranslatedGenerationIssue = (
  message: string,
  target: GenerationIssue['target'],
  fix?: GenerationIssueFix
): GenerationIssue => ({
  id: `${target.section}:${target.field ?? ''}:${message}`,
  message,
  target,
  ...(fix ? { fix } : {}),
});
