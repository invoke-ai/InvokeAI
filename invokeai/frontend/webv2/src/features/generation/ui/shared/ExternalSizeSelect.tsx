/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { createListCollection } from '@chakra-ui/react';
import {
  findExternalResolutionPreset,
  getExternalAspectRatioSizes,
  getExternalResolutionPresets,
} from '@features/generation/core/externalProvider';
import { isAspectRatioId } from '@features/generation/core/settings';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type SizeSettings = Pick<GenerateSettings, 'externalImageSize' | 'height' | 'width'>;

interface ExternalSizeSelectProps {
  model: GenerateModelConfig | undefined;
  settings: SizeSettings;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const presetKey = (preset: { aspect_ratio: string; image_size: string }): string =>
  `${preset.aspect_ratio}|${preset.image_size}`;

/** The size patch a provider entry implies; its ratio is locked so the width/height fields keep it. */
const sizePatch = (ratio: string, width: number, height: number): Partial<GenerateSettings> => ({
  aspectRatioId: isAspectRatioId(ratio) ? ratio : 'Free',
  aspectRatioIsLocked: true,
  aspectRatioValue: height > 0 ? width / height : 1,
  height,
  width,
});

/**
 * A hosted provider's own size table: named sizes per ratio (Gemini 3's `1K`/`2K`/`4K`), or one fixed size per
 * ratio. Renders nothing for free-form providers and local models. A size typed by hand shows as custom.
 */
export const ExternalSizeSelect = ({ model, onCommit, settings }: ExternalSizeSelectProps) => {
  const { t } = useTranslation();
  // One table per model: the policy reads fresh arrays off the record, so the record is the memo key.
  const { collection, presets, ratioSizes } = useMemo(() => {
    const presets = getExternalResolutionPresets(model);
    const ratioSizes = getExternalAspectRatioSizes(model);

    return {
      collection: createListCollection({
        items:
          presets.length > 0
            ? presets.map((preset) => ({ label: preset.label, value: presetKey(preset) }))
            : ratioSizes.map(([ratio, size]) => ({ label: `${ratio} (${size.width}×${size.height})`, value: ratio })),
      }),
      presets,
      ratioSizes,
    };
  }, [model]);

  if (presets.length === 0 && ratioSizes.length === 0) {
    return null;
  }

  const currentPreset = presets.length > 0 ? findExternalResolutionPreset(settings, model) : null;
  const currentRatio =
    presets.length > 0
      ? null
      : (ratioSizes.find(([, size]) => size.width === settings.width && size.height === settings.height)?.[0] ?? null);
  const value = currentPreset ? [presetKey(currentPreset)] : currentRatio ? [currentRatio] : [];

  return (
    <Field label={t('widgets.generate.providerSize')} helpText={t('widgets.generate.providerSizeHelp')}>
      <Select
        aria-label={t('widgets.generate.providerSize')}
        collection={collection}
        value={value}
        valueTextProps={{ placeholder: t('widgets.generate.providerSizeCustom') }}
        onValueChange={({ value: next }) => {
          const key = next[0];

          if (key === undefined) {
            return;
          }

          if (presets.length > 0) {
            const preset = presets.find((candidate) => presetKey(candidate) === key);

            if (preset) {
              onCommit({
                ...sizePatch(preset.aspect_ratio, preset.width, preset.height),
                externalImageSize: preset.image_size,
              });
            }
            return;
          }

          const entry = ratioSizes.find(([ratio]) => ratio === key);

          if (entry) {
            onCommit({ ...sizePatch(entry[0], entry[1].width, entry[1].height), externalImageSize: null });
          }
        }}
      />
    </Field>
  );
};
