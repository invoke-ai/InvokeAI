import type { GenerateSettings, MainModelConfig, VaeModelConfig } from '@workbench/generation/types';

import { Stack, Text } from '@chakra-ui/react';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';
import { GenerateDimensionFields } from './GenerateDimensionFields';
import { GenerateModelFields } from './GenerateModelFields';
import { GeneratePromptFields } from './GeneratePromptFields';

interface GenerateSettingsFormProps {
  isLoadingModels: boolean;
  loadError: string | null;
  settings: GenerateSettings;
  selectedModel: MainModelConfig | undefined;
  supportedModels: MainModelConfig[];
  vaeModels: VaeModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

export const GenerateSettingsForm = ({
  isLoadingModels,
  loadError,
  onCommitSettings,
  selectedModel,
  settings,
  supportedModels,
  vaeModels,
}: GenerateSettingsFormProps) => {
  const commit = (patch: Partial<GenerateSettings>) => {
    onCommitSettings({ ...settings, ...patch });
  };

  return (
    <Stack gap="1" px={1}>
      <GeneratePromptFields settings={settings} onCommit={commit} />

      <GenerateModelFields
        selectedModel={selectedModel}
        settings={settings}
        vaeModels={vaeModels}
        onCommit={commit}
        onCommitSettings={onCommitSettings}
      />

      <GenerateDimensionFields selectedModel={selectedModel} settings={settings} onCommit={commit} />
      <GenerateAdvancedFields selectedModel={selectedModel} settings={settings} onCommit={commit} />
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
      ) : selectedModel ? null : (
        <Text color="fg.error" fontSize="2xs">
          Select a model to enable invocation.
        </Text>
      )}
    </Stack>
  );
};
