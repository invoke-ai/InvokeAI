import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@workbench/generation/types';

import { Stack, Text } from '@chakra-ui/react';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';
import { GenerateComponentsSection } from './GenerateComponentsSection';
import { GenerateConceptsSection } from './GenerateConceptsSection';
import { GenerateDimensionFields } from './GenerateDimensionFields';
import { GenerateModelFields } from './GenerateModelFields';
import { GeneratePromptFields } from './promptFields';

interface GenerateSettingsFormProps {
  isLoadingModels: boolean;
  loadError: string | null;
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
  supportedModels: GenerateModelConfig[];
  vaeModels: VaeModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
}

export const GenerateSettingsForm = ({
  isLoadingModels,
  loadError,
  loraModels,
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
      <GeneratePromptFields selectedModel={selectedModel} settings={settings} onCommit={commit} />

      <GenerateDimensionFields selectedModel={selectedModel} settings={settings} onCommit={commit} />

      <GenerateModelFields
        selectedModel={selectedModel}
        settings={settings}
        vaeModels={vaeModels}
        onCommit={commit}
        onCommitSettings={onCommitSettings}
      />

      <GenerateConceptsSection
        loraModels={loraModels}
        selectedModel={selectedModel}
        settings={settings}
        onCommit={commit}
      />

      <GenerateComponentsSection selectedModel={selectedModel} settings={settings} onCommit={commit} />

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
          No supported generation models were found on the backend.
        </Text>
      ) : selectedModel ? null : (
        <Text color="fg.error" fontSize="2xs">
          Select a model to enable invocation.
        </Text>
      )}
    </Stack>
  );
};
