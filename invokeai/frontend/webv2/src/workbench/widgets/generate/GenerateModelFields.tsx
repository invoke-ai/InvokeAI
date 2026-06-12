import { HStack, NativeSelect, NumberInput, Stack } from '@chakra-ui/react';
import { DicesIcon } from 'lucide-react';
import type { ChangeEvent } from 'react';

import { Button, IconButton } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { ModelSelect } from '../../components/ModelSelect';
import { Tooltip } from '../../components/ui/Tooltip';
import { getSettingsWithModelDefaults, isSupportedGenerateModel } from '../../generation/graph';
import { isKnownScheduler, isMainModelConfig, SCHEDULER_OPTIONS, SEED_MAX } from '../../generation/settings';
import type { GenerateSettings, MainModelConfig, VaeModelConfig } from '../../generation/types';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateModelFieldsProps {
  settings: GenerateSettings;
  selectedModel: MainModelConfig | undefined;
  vaeModels: VaeModelConfig[];
  onCommit: (patch: Partial<GenerateSettings>) => void;
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

export const GenerateModelFields = ({
  onCommit,
  onCommitSettings,
  selectedModel,
  settings,
  vaeModels,
}: GenerateModelFieldsProps) => {
  const commitNumber = (key: 'cfgScale' | 'seed' | 'steps', value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }

    onCommit({ [key]: value });
  };

  const updateNumberInput =
    (key: 'cfgScale' | 'seed' | 'steps') =>
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      commitNumber(key, valueAsNumber);
    };

  const selectModel = (model: MainModelConfig) => {
    // The model's configured default VAE (a model key) only applies if that VAE is installed.
    const defaultVaeKey = model.default_settings?.vae;
    const defaultVae = defaultVaeKey
      ? (vaeModels.find((vae) => vae.key === defaultVaeKey && vae.base === model.base) ?? null)
      : null;

    onCommitSettings({ ...getSettingsWithModelDefaults(settings, model), vae: defaultVae });
  };

  return (
    <GenerateCollapsibleSection label="Model" defaultOpen>
      <Stack gap="2" p="2">
        <Field label="Model">
          <ModelSelect
            filter={(model) => isMainModelConfig(model) && isSupportedGenerateModel(model)}
            modelTypes={['main']}
            placeholder="Select a model…"
            value={selectedModel?.key ?? null}
            size="xs"
            onChange={(model) => {
              if (isMainModelConfig(model) && isSupportedGenerateModel(model)) {
                selectModel(model);
              }
            }}
          />
        </Field>
        <HStack gap="2">
          <Field label="Steps">
            <NumberInput.Root
              size="xs"
              value={String(settings.steps)}
              onValueChange={updateNumberInput('steps')}
              min={1}
              allowMouseWheel
            >
              <NumberInput.Input />
              <NumberInput.Control />
            </NumberInput.Root>
          </Field>
          <Field label="CFG">
            <NumberInput.Root
              size="xs"
              value={String(settings.cfgScale)}
              onValueChange={updateNumberInput('cfgScale')}
              min={0}
              max={10}
              step={0.5}
            >
              <NumberInput.Input />
              <NumberInput.Control />
            </NumberInput.Root>
          </Field>
          <Field label="Scheduler">
            <NativeSelect.Root size="xs">
              <NativeSelect.Field
                aria-label="Scheduler"
                value={settings.scheduler}
                onChange={(event: ChangeEvent<HTMLSelectElement>) => onCommit({ scheduler: event.currentTarget.value })}
              >
                {isKnownScheduler(settings.scheduler) ? null : (
                  <option value={settings.scheduler}>{settings.scheduler}</option>
                )}
                {SCHEDULER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </NativeSelect.Field>
              <NativeSelect.Indicator />
            </NativeSelect.Root>
          </Field>
        </HStack>
        <Field label="Seed">
          <HStack gap="2">
            <NumberInput.Root
              size="xs"
              value={String(settings.seed)}
              onValueChange={updateNumberInput('seed')}
              min={0}
              max={SEED_MAX}
              w="full"
              disabled={settings.shouldRandomizeSeed}
            >
              <NumberInput.Input />
            </NumberInput.Root>
            <Tooltip content="Shuffle seed">
              <IconButton
                aria-label="Shuffle seed"
                disabled={settings.shouldRandomizeSeed}
                size="xs"
                variant="outline"
                onClick={() => onCommit({ seed: Math.floor(Math.random() * SEED_MAX) })}
              >
                <DicesIcon />
              </IconButton>
            </Tooltip>
            <Button
              size="xs"
              variant={settings.shouldRandomizeSeed ? 'solid' : 'outline'}
              onClick={() => onCommit({ shouldRandomizeSeed: !settings.shouldRandomizeSeed })}
            >
              Random
            </Button>
          </HStack>
        </Field>
      </Stack>
    </GenerateCollapsibleSection>
  );
};
