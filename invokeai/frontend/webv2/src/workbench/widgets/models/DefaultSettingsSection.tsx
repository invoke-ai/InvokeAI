import type { AnyModelDefaultSettings, ModelConfig } from '@workbench/models/types';

import { Grid, HStack, Icon, NativeSelect, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import { FieldLabel } from '@workbench/components/ui/Field';
import { Panel } from '@workbench/components/ui/Panel';
import { updateModel } from '@workbench/models/api';
import { replaceModelInStore } from '@workbench/models/modelsStore';
import { loraDefaultSettingsSchema, mainDefaultSettingsSchema } from '@workbench/models/schemas';
import { MoveHorizontalIcon } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';

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

export const supportsDefaultSettings = (model: ModelConfig): boolean =>
  model.type === 'main' || model.type === 'lora' || CONTROL_ADAPTER_TYPES.has(model.type);

interface FieldSpec {
  key: keyof AnyModelDefaultSettings;
  label: string;
  /** Render the control; disabled (showing the inherited value) until customized. */
  control: (value: unknown, setValue: (value: unknown) => void, disabled: boolean) => ReactNode;
  /** Value used when the toggle is switched on; also previewed while off. */
  defaultValue: unknown;
  /** What applies while the toggle is off, e.g. "App default: 30 steps". */
  inheritLabel: string;
}

const numberControl =
  (props: { max?: number; min?: number; step?: number }) =>
  (value: unknown, setValue: (value: unknown) => void, disabled: boolean) => (
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

const selectControl =
  (options: string[]) => (value: unknown, setValue: (value: unknown) => void, disabled: boolean) => (
    <NativeSelect.Root disabled={disabled} size="sm">
      <NativeSelect.Field
        value={typeof value === 'string' ? value : ''}
        onChange={(event) => setValue(event.currentTarget.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </NativeSelect.Field>
      <NativeSelect.Indicator />
    </NativeSelect.Root>
  );

const MAIN_FIELDS: FieldSpec[] = [
  {
    control: selectControl(SCHEDULERS),
    defaultValue: 'euler_a',
    inheritLabel: 'App default: euler_a',
    key: 'scheduler',
    label: 'Scheduler',
  },
  {
    control: numberControl({ max: 10000, min: 1 }),
    defaultValue: 30,
    inheritLabel: 'App default: 30',
    key: 'steps',
    label: 'Steps',
  },
  {
    control: numberControl({ max: 200, min: 1, step: 0.5 }),
    defaultValue: 7,
    inheritLabel: 'App default: 7',
    key: 'cfg_scale',
    label: 'CFG Scale',
  },
  {
    control: numberControl({ max: 0.99, min: 0, step: 0.05 }),
    defaultValue: 0,
    inheritLabel: 'App default: 0 (off)',
    key: 'cfg_rescale_multiplier',
    label: 'CFG Rescale',
  },
  {
    control: numberControl({ max: 20, min: 1, step: 0.5 }),
    defaultValue: 4,
    inheritLabel: 'Uses the generator default',
    key: 'guidance',
    label: 'Guidance',
  },
  {
    control: numberControl({ max: 8192, min: 64, step: 8 }),
    defaultValue: 1024,
    inheritLabel: 'App default: 1024',
    key: 'width',
    label: 'Width',
  },
  {
    control: numberControl({ max: 8192, min: 64, step: 8 }),
    defaultValue: 1024,
    inheritLabel: 'App default: 1024',
    key: 'height',
    label: 'Height',
  },
  {
    control: selectControl(['fp16', 'fp32']),
    defaultValue: 'fp16',
    inheritLabel: 'App default: fp16',
    key: 'vae_precision',
    label: 'VAE Precision',
  },
];

const LORA_FIELDS: FieldSpec[] = [
  {
    control: numberControl({ max: 10, min: -10, step: 0.05 }),
    defaultValue: 0.75,
    inheritLabel: 'App default: 0.75',
    key: 'weight',
    label: 'Weight',
  },
];

const CONTROL_ADAPTER_FIELDS: FieldSpec[] = [
  {
    control: selectControl(PREPROCESSORS),
    defaultValue: 'canny_edge_detection',
    inheritLabel: 'Chosen automatically per image',
    key: 'preprocessor',
    label: 'Preprocessor',
  },
];

const getFieldsForModel = (model: ModelConfig): FieldSpec[] => {
  if (model.type === 'main') {
    return MAIN_FIELDS;
  }

  if (model.type === 'lora') {
    return LORA_FIELDS;
  }

  return CONTROL_ADAPTER_FIELDS;
};

const validateDefaults = (model: ModelConfig, settings: AnyModelDefaultSettings): string | null => {
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

    return result.success ? null : (result.error.issues[0]?.message ?? 'Invalid default settings.');
  }

  if (model.type === 'lora') {
    const result = loraDefaultSettingsSchema.safeParse({ weight: settings.weight ?? null });

    return result.success ? null : (result.error.issues[0]?.message ?? 'Invalid default settings.');
  }

  return null;
};

export const DefaultSettingsSection = ({
  model,
  onError,
  onSaved,
}: {
  model: ModelConfig;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const fields = useMemo(() => getFieldsForModel(model), [model]);
  const [settings, setSettings] = useState<AnyModelDefaultSettings>(() => ({ ...model.default_settings }));
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const isDirty = useMemo(() => {
    const saved = model.default_settings ?? {};

    return fields.some((field) => (settings[field.key] ?? null) !== (saved[field.key] ?? null));
  }, [fields, model.default_settings, settings]);

  const setFieldValue = (key: keyof AnyModelDefaultSettings, value: unknown) => {
    setSettings((current) => ({ ...current, [key]: value as never }));
    setError(null);
  };

  const handleSave = async () => {
    const validationError = validateDefaults(model, settings);

    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSaving(true);

    try {
      replaceModelInStore(await updateModel(model.key, { default_settings: settings }));
      onSaved();
    } catch (saveError) {
      onError(saveError instanceof Error ? saveError.message : 'Failed to save default settings.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Stack gap="3">
      <HStack justify="space-between">
        <Stack gap="0.5">
          <FieldLabel>Default Settings</FieldLabel>
          <Text color="fg.subtle" fontSize="2xs">
            Applied automatically when this model is selected. Off = use the app default.
          </Text>
        </Stack>
        <Button disabled={!isDirty} loading={isSaving} size="xs" variant="solid" onClick={() => void handleSave()}>
          Save Defaults
        </Button>
      </HStack>
      {error ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {error}
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
                  {field.label}
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
                  <Switch.Label srOnly>{`Customize ${field.label} for this model`}</Switch.Label>
                </Switch.Root>
              </HStack>
              {field.control(
                isEnabled ? value : field.defaultValue,
                (nextValue) => setFieldValue(field.key, nextValue),
                !isEnabled
              )}
              <Text color="fg.subtle" fontSize="2xs">
                {isEnabled ? 'Customized for this model' : field.inheritLabel}
              </Text>
            </Panel>
          );
        })}
      </Grid>
    </Stack>
  );
};
