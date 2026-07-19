import type { GenerationModelCatalogItem, PromptHistoryItem } from '@features/generation/contracts';
import type { GenerateSettings } from '@features/generation/core/types';
import type { ComponentType, ReactNode } from 'react';

import { createContext, use } from 'react';

export interface GenerationModelSelectProps {
  className?: string;
  disabled?: boolean;
  excludeKeys?: ReadonlySet<string>;
  filter?: (model: GenerationModelCatalogItem) => boolean;
  id?: string;
  invalid?: boolean;
  isClearable?: boolean;
  modelTypes: string[];
  onChange: (model: GenerationModelCatalogItem | null) => void;
  placeholder?: string;
  showManagerButton?: boolean;
  size?: 'xs' | 'sm' | 'md';
  value: string | null;
}

export interface GenerationSelectedImage {
  imageName: string;
  imageUrl: string;
  thumbnailUrl: string;
}

export interface GenerationUiAdapter {
  CanvasCompositingSection: ComponentType;
  ModelSelect: ComponentType<GenerationModelSelectProps>;
  activeProjectId: string;
  canvasValues: Record<string, unknown>;
  ensureModelsLoaded(): void;
  generateValues: Record<string, unknown>;
  getModelBaseColorPalette(base: string): string;
  getModelBaseLabel(base: string): string;
  invocationSourceId: string;
  models: readonly GenerationModelCatalogItem[];
  modelsError: string | null;
  modelsStatus: 'error' | 'idle' | 'loaded' | 'loading';
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    reportError(error: { area: string; message: string; namespace: 'generation'; projectId?: string }): void;
  };
  patchCanvasValues(values: Record<string, unknown>): void;
  patchGenerateSettings(values: Partial<GenerateSettings>, projectId?: string): void;
  promptHistory: readonly PromptHistoryItem[];
  clearPromptHistory(): void;
  removePromptFromHistory(prompt: PromptHistoryItem): void;
  selectedGalleryImage: GenerationSelectedImage | null;
  showPromptSyntaxHighlighting: boolean;
  touchGalleryImages(): void;
}

const GenerationUiContext = createContext<GenerationUiAdapter | null>(null);

export const GenerationUiProvider = ({ adapter, children }: { adapter: GenerationUiAdapter; children: ReactNode }) => (
  <GenerationUiContext value={adapter}>{children}</GenerationUiContext>
);

export const useGenerationUi = (): GenerationUiAdapter => {
  const adapter = use(GenerationUiContext);

  if (!adapter) {
    throw new Error('Generation UI requires an App-composed GenerationUiProvider.');
  }

  return adapter;
};

export const GenerationModelSelect = (props: GenerationModelSelectProps) => {
  const { ModelSelect } = useGenerationUi();
  return <ModelSelect {...props} />;
};
