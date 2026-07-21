/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { AnyModelDefaultSettings, ModelConfig } from '@features/models/core/types';
import type { TFunction } from 'i18next';

import { createListCollection, Grid, HStack, Icon, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { loraDefaultSettingsSchema, mainDefaultSettingsSchema } from '@features/models/core/schemas';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { Button, Combobox, FieldLabel, Panel, Select } from '@platform/ui';
import { MoveHorizontalIcon } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Per-model generation defaults ("use these settings when this model is
 * selected"). Every field is individually toggleable: off = inherit the app
 * default (stored as null). Validation runs through zod on save.
 */

const SCHEDULERS = [
  'ddim',
  'ddpm',
  'deis',
  'deis_k',
  'dpmpp_2s',
  'dpmpp_2s_k',
  'dpmpp_2m',
  'dpmpp_2m_k',
  'dpmpp_2m_sde',
  'dpmpp_2m_sde_k',
  'dpmpp_3m',
  'dpmpp_3m_k',
  'dpmpp_sde',
  'dpmpp_sde_k',
  'euler',
  'euler_k',
  'euler_a',
  'heun',
  'heun_k',
  'kdpm_2',
  'kdpm_2_k',
  'kdpm_2_a',
  'kdpm_2_a_k',
  'lcm',
  'lms',
  'lms_k',
  'pndm',
  'tcd',
  'unipc',
  'unipc_k',
];
const SCHEDULER_OPTIONS = SCHEDULERS.map((scheduler) => ({ label: scheduler, value: scheduler }));

const CONTROL_ADAPTER_TYPES = new Set(['controlnet', 't2i_adapter', 'control_lora']);

const PREPROCESSORS = [
  'canny_edge_detection',
  'color_map',
  'content_shuffle',
  'depth_anything_depth_estimation',
  'dw_openpose_detection',
  'hed_edge_detection',
  'lineart_anime_edge_detection',
  'lineart_edge_detection',
  'mediapipe_face_detection',
  'mlsd_detection',
  'normal_map',
  'pidi_edge_detection',
  'tile',
];

type DefaultSettingsModel = Pick<ModelConfig, 'default_settings' | 'key' | 'type'>;

interface DefaultSettingsDraft {
  modelKey: string;
  settings: AnyModelDefaultSettings;
  source: AnyModelDefaultSettings | null | undefined;
}

export const supportsDefaultSettings = (model: Pick<ModelConfig, 'type'>): boolean =>
  model.type === 'main' || model.type === 'lora' || CONTROL_ADAPTER_TYPES.has(model.type);

interface FieldSpec {
  key: keyof AnyModelDefaultSettings;
  labelKey: string;
  /** Render the control; disabled (showing the inherited value) until customized. */
  control: (value: unknown, setValue: (value: unknown) => void, disabled: boolean, label: string) => ReactNode;
  /** Value used when the toggle is switched on; also previewed while off. */
  defaultValue: unknown;
  /** Translation key for what applies while the toggle is off. */
  inheritLabelKey: string;
}

const numberControl =
  (props: { max?: number; min?: number; step?: number }) =>
  (value: unknown, setValue: (value: unknown) => void, disabled: boolean, _label: string) => (
    <NumberInput.Root
      disabled={disabled}
      max={props.max}
      min={props.min}
      position="relative"
      size="sm"
      step={props.step ?? 1}
      value={typeof value === 'number' ? String(value) : ''}
      w="full"
      onValueChange={(details) => {
        // Empty/partial input is transient; only commit finite numbers so the
        // field never silently flips back to "off".
        if (Number.isFinite(details.valueAsNumber)) {
          setValue(details.valueAsNumber);
        }
      }}
    >
      <NumberInput.Control />
      {/* Drag horizontally on the handle to scrub the value. */}
      <NumberInput.Scrubber
        alignItems="center"
        bottom="0"
        cursor="ew-resize"
        display="flex"
        left="2"
        position="absolute"
        top="0"
        zIndex={1}
      >
        <Icon as={MoveHorizontalIcon} boxSize="3" color="fg.subtle" />
      </NumberInput.Scrubber>
      <NumberInput.Input ps="7" />
    </NumberInput.Root>
  );

const selectControl = (options: string[]) => {
  // Called only at module scope, so the collection is built once per field spec.
  const collection = createListCollection({
    items: options.map((option) => ({ label: option, value: option })),
  });

  return (value: unknown, setValue: (value: unknown) => void, disabled: boolean, label: string) => (
    <Select
      aria-label={label}
      collection={collection}
      disabled={disabled}
      size="sm"
      value={typeof value === 'string' ? [value] : []}
      onValueChange={({ value: next }) => {
        const nextValue = next[0];

        if (nextValue !== undefined) {
          setValue(nextValue);
        }
      }}
    />
  );
};

const schedulerControl =
  (options: { label: string; value: string }[]) =>
  (value: unknown, setValue: (value: unknown) => void, disabled: boolean, label: string) => (
    <Combobox
      aria-label={label}
      disabled={disabled}
      options={options}
      size="sm"
      value={typeof value === 'string' ? value : null}
      onValueChange={setValue}
    />
  );

const MAIN_FIELDS: FieldSpec[] = [
  {
    control: schedulerControl(SCHEDULER_OPTIONS),
    defaultValue: 'euler_a',
    inheritLabelKey: 'models.defaultFieldInherited.scheduler',
    key: 'scheduler',
    labelKey: 'models.defaultFields.scheduler',
  },
  {
    control: numberControl({ max: 10000, min: 1 }),
    defaultValue: 30,
    inheritLabelKey: 'models.defaultFieldInherited.steps',
    key: 'steps',
    labelKey: 'models.defaultFields.steps',
  },
  {
    control: numberControl({ max: 200, min: 1, step: 0.5 }),
    defaultValue: 7,
    inheritLabelKey: 'models.defaultFieldInherited.cfgScale',
    key: 'cfg_scale',
    labelKey: 'models.defaultFields.cfgScale',
  },
  {
    control: numberControl({ max: 0.99, min: 0, step: 0.05 }),
    defaultValue: 0,
    inheritLabelKey: 'models.defaultFieldInherited.cfgRescale',
    key: 'cfg_rescale_multiplier',
    labelKey: 'models.defaultFields.cfgRescale',
  },
  {
    control: numberControl({ max: 20, min: 1, step: 0.5 }),
    defaultValue: 4,
    inheritLabelKey: 'models.defaultFieldInherited.guidance',
    key: 'guidance',
    labelKey: 'models.defaultFields.guidance',
  },
  {
    control: numberControl({ max: 8192, min: 64, step: 8 }),
    defaultValue: 1024,
    inheritLabelKey: 'models.defaultFieldInherited.width',
    key: 'width',
    labelKey: 'models.defaultFields.width',
  },
  {
    control: numberControl({ max: 8192, min: 64, step: 8 }),
    defaultValue: 1024,
    inheritLabelKey: 'models.defaultFieldInherited.height',
    key: 'height',
    labelKey: 'models.defaultFields.height',
  },
  {
    control: selectControl(['fp16', 'fp32']),
    defaultValue: 'fp16',
    inheritLabelKey: 'models.defaultFieldInherited.vaePrecision',
    key: 'vae_precision',
    labelKey: 'models.defaultFields.vaePrecision',
  },
];

const LORA_FIELDS: FieldSpec[] = [
  {
    control: numberControl({ max: 10, min: -10, step: 0.05 }),
    defaultValue: 0.75,
    inheritLabelKey: 'models.defaultFieldInherited.weight',
    key: 'weight',
    labelKey: 'models.defaultFields.weight',
  },
];

const CONTROL_ADAPTER_FIELDS: FieldSpec[] = [
  {
    control: selectControl(PREPROCESSORS),
    defaultValue: 'canny_edge_detection',
    inheritLabelKey: 'models.defaultFieldInherited.preprocessor',
    key: 'preprocessor',
    labelKey: 'models.defaultFields.preprocessor',
  },
];

const getFieldsForModel = (model: Pick<ModelConfig, 'type'>): FieldSpec[] => {
  if (model.type === 'main') {
    return MAIN_FIELDS;
  }

  if (model.type === 'lora') {
    return LORA_FIELDS;
  }

  return CONTROL_ADAPTER_FIELDS;
};

const validateDefaults = (
  model: Pick<ModelConfig, 'type'>,
  settings: AnyModelDefaultSettings,
  t: TFunction
): string | null => {
  if (model.type === 'main') {
    const result = mainDefaultSettingsSchema.safeParse({
      cfgRescaleMultiplier: settings.cfg_rescale_multiplier ?? null,
      cfgScale: settings.cfg_scale ?? null,
      guidance: settings.guidance ?? null,
      height: settings.height ?? null,
      scheduler: settings.scheduler ?? null,
      steps: settings.steps ?? null,
      vaePrecision: settings.vae_precision ?? null,
      width: settings.width ?? null,
    });

    return result.success ? null : (result.error.issues[0]?.message ?? t('models.invalidDefaultSettings'));
  }

  if (model.type === 'lora') {
    const result = loraDefaultSettingsSchema.safeParse({ weight: settings.weight ?? null });

    return result.success ? null : (result.error.issues[0]?.message ?? t('models.invalidDefaultSettings'));
  }

  return null;
};

export const DefaultSettingsSection = ({
  model,
  onError,
  onSaved,
}: {
  model: DefaultSettingsModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const fields = useMemo(() => getFieldsForModel(model), [model]);
  const [draft, setDraft] = useState<DefaultSettingsDraft>(() => ({
    modelKey: model.key,
    settings: { ...model.default_settings },
    source: model.default_settings,
  }));
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const isDraftCurrent = draft.modelKey === model.key && draft.source === model.default_settings;
  const savedSettingsDraft = useMemo(() => ({ ...model.default_settings }), [model.default_settings]);
  const settings = isDraftCurrent ? draft.settings : savedSettingsDraft;
  const visibleError = isDraftCurrent ? error : null;
  const visibleIsSaving = isDraftCurrent ? isSaving : false;

  const isDirty = useMemo(() => {
    const saved = model.default_settings ?? {};

    return fields.some((field) => (settings[field.key] ?? null) !== (saved[field.key] ?? null));
  }, [fields, model.default_settings, settings]);

  const setFieldValue = (key: keyof AnyModelDefaultSettings, value: unknown) => {
    setDraft({
      modelKey: model.key,
      settings: { ...settings, [key]: value as never },
      source: model.default_settings,
    });
    setError(null);
  };

  const handleSave = async () => {
    const validationError = validateDefaults(model, settings, t);

    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSaving(true);

    try {
      replaceModelInStore(await updateModel(model.key, { default_settings: settings }));
      onSaved();
    } catch (saveError) {
      onError(saveError instanceof Error ? saveError.message : t('models.failedToSaveDefaults'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Stack gap="3">
      <HStack justify="space-between">
        <Stack gap="0.5">
          <FieldLabel>{t('models.defaultSettings')}</FieldLabel>
          <Text color="fg.subtle" fontSize="2xs">
            {t('models.defaultSettingsHelp')}
          </Text>
        </Stack>
        <Button
          disabled={!isDirty}
          loading={visibleIsSaving}
          size="xs"
          variant="solid"
          onClick={() => void handleSave()}
        >
          {t('models.saveDefaults')}
        </Button>
      </HStack>
      {visibleError ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {visibleError}
        </Text>
      ) : null}
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
        {fields.map((field) => {
          const value = settings[field.key] ?? null;
          const isEnabled = value !== null && value !== undefined;

          return (
            <Panel key={field.key} gap="2" p="2.5" tone="control">
              <HStack justify="space-between">
                <Text fontSize="2xs" fontWeight="600" textTransform="uppercase">
                  {t(field.labelKey)}
                </Text>
                <Switch.Root
                  checked={isEnabled}
                  colorPalette="accent"
                  size="xs"
                  onCheckedChange={(event) => setFieldValue(field.key, event.checked ? field.defaultValue : null)}
                >
                  <Switch.HiddenInput />
                  <Switch.Control>
                    <Switch.Thumb />
                  </Switch.Control>
                  <Switch.Label srOnly>{t('models.customizeDefaultField', { field: t(field.labelKey) })}</Switch.Label>
                </Switch.Root>
              </HStack>
              {field.control(
                isEnabled ? value : field.defaultValue,
                (nextValue) => setFieldValue(field.key, nextValue),
                !isEnabled,
                t(field.labelKey)
              )}
              <Text color="fg.subtle" fontSize="2xs">
                {isEnabled ? t('models.customizedForThisModel') : t(field.inheritLabelKey)}
              </Text>
            </Panel>
          );
        })}
      </Grid>
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
