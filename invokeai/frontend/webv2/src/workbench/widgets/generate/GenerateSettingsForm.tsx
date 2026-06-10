import { HStack, Input, NativeSelect, Stack, Text, Textarea } from '@chakra-ui/react';
import type { ChangeEvent } from 'react';

import { Button } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { getSettingsWithModelDefaults } from '../../generation/graph';
import type { GenerateSettings, MainModelConfig } from '../../generation/types';

interface GenerateSettingsFormProps {
  isLoadingModels: boolean;
  loadError: string | null;
  settings: GenerateSettings;
  supportedModels: MainModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

export const GenerateSettingsForm = ({
  isLoadingModels,
  loadError,
  settings,
  supportedModels,
  onCommitSettings,
}: GenerateSettingsFormProps) => {
  const updateText = (key: 'negativePrompt' | 'positivePrompt') => (event: ChangeEvent<HTMLTextAreaElement>) => {
    onCommitSettings({ ...settings, [key]: event.currentTarget.value });
  };

  const updateNumber =
    (key: 'cfgScale' | 'height' | 'seed' | 'steps' | 'width') => (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.currentTarget.value);

      if (!Number.isFinite(value)) {
        return;
      }

      onCommitSettings({ ...settings, [key]: value });
    };

  return (
    <Stack gap="3">
      <Field label="Prompt">
        <Textarea
          aria-label="Positive prompt"
          minH="6rem"
          resize="vertical"
          size="xs"
          fontFamily="mono"
          value={settings.positivePrompt}
          onChange={updateText('positivePrompt')}
        />
      </Field>
      <Field label="Negative prompt">
        <Textarea
          aria-label="Negative prompt"
          minH="3.5rem"
          resize="vertical"
          size="xs"
          fontFamily="mono"
          value={settings.negativePrompt}
          onChange={updateText('negativePrompt')}
        />
      </Field>
      <Field label="Model">
        <NativeSelect.Root disabled={supportedModels.length === 0} size="sm">
          <NativeSelect.Field
            aria-label="Model"
            value={settings.modelKey}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => {
              const model = supportedModels.find((candidate) => candidate.key === event.currentTarget.value);

              if (!model) {
                return;
              }

              onCommitSettings({
                ...getSettingsWithModelDefaults(settings, model),
                negativePrompt: settings.negativePrompt,
                positivePrompt: settings.positivePrompt,
              });
            }}
          >
            {supportedModels.map((model) => (
              <option key={model.key} value={model.key}>
                {model.name} ({model.base})
              </option>
            ))}
          </NativeSelect.Field>
          <NativeSelect.Indicator />
        </NativeSelect.Root>
      </Field>
      <HStack gap="2">
        <Field label="Width">
          <Input min="64" size="sm" step="8" type="number" value={settings.width} onChange={updateNumber('width')} />
        </Field>
        <Field label="Height">
          <Input min="64" size="sm" step="8" type="number" value={settings.height} onChange={updateNumber('height')} />
        </Field>
      </HStack>
      <HStack gap="2">
        <Field label="Steps">
          <Input min="1" size="sm" type="number" value={settings.steps} onChange={updateNumber('steps')} />
        </Field>
        <Field label="CFG">
          <Input
            min="0"
            size="sm"
            step="0.5"
            type="number"
            value={settings.cfgScale}
            onChange={updateNumber('cfgScale')}
          />
        </Field>
      </HStack>
      <Field label="Scheduler">
        <Input
          aria-label="Scheduler"
          size="sm"
          value={settings.scheduler}
          onChange={(event: ChangeEvent<HTMLInputElement>) =>
            onCommitSettings({ ...settings, scheduler: event.currentTarget.value })
          }
        />
      </Field>
      <Field label="Seed">
        <HStack gap="2">
          <Input
            disabled={settings.shouldRandomizeSeed}
            min="0"
            size="sm"
            type="number"
            value={settings.seed}
            onChange={updateNumber('seed')}
          />
          <Button
            size="xs"
            variant={settings.shouldRandomizeSeed ? 'solid' : 'outline'}
            onClick={() => onCommitSettings({ ...settings, shouldRandomizeSeed: !settings.shouldRandomizeSeed })}
          >
            Random
          </Button>
        </HStack>
      </Field>
      {isLoadingModels ? (
        <Text color="fg.subtle" fontSize="2xs">
          Loading backend models...
        </Text>
      ) : loadError ? (
        <Text color="fg.error" fontSize="2xs">
          {loadError}
        </Text>
      ) : supportedModels.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          No SD/SDXL main models were found on the backend.
        </Text>
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          Current settings compile into an immutable Generate graph snapshot when you invoke.
        </Text>
      )}
    </Stack>
  );
};
