import type {
  ExternalApiModelConfig,
  ExternalModelCapabilities,
  ExternalModelPanelControl,
  ExternalModelPanelSchema,
  ExternalPanelControlName,
} from 'services/api/types';

type ExternalPanelName = keyof ExternalModelPanelSchema;

const buildExternalPanelSchemaFromCapabilities = (
  capabilities: ExternalModelCapabilities
): ExternalModelPanelSchema => ({
  prompts: [...(capabilities.supports_reference_images ? [{ name: 'reference_images' as const }] : [])],
  image: [{ name: 'dimensions' }, ...(capabilities.supports_seed ? [{ name: 'seed' as const }] : [])],
  generation: [],
});

const getExternalPanelSchema = (modelConfig: ExternalApiModelConfig): ExternalModelPanelSchema =>
  modelConfig.panel_schema ?? buildExternalPanelSchemaFromCapabilities(modelConfig.capabilities);

export const getExternalPanelControl = (
  modelConfig: ExternalApiModelConfig,
  panel: ExternalPanelName,
  controlName: ExternalPanelControlName
): ExternalModelPanelControl | null =>
  getExternalPanelSchema(modelConfig)[panel].find((control) => control.name === controlName) ?? null;

export const hasExternalPanelControl = (
  modelConfig: ExternalApiModelConfig,
  panel: ExternalPanelName,
  controlName: ExternalPanelControlName
): boolean => getExternalPanelControl(modelConfig, panel, controlName) !== null;
