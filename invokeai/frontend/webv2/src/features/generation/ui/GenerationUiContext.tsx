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

/**
 * Generation's UI port, grouped into sub-ports by backing concern. The context
 * is a dependency-direction port (the feature may not import workbench), not a
 * test seam; no second adapter is expected.
 */
export interface GenerationUiAdapter {
  CanvasCompositingSection: ComponentType;
  gallery: {
    selectedImage: GenerationSelectedImage | null;
    touchImages(): void;
  };
  models: {
    ModelSelect: ComponentType<GenerationModelSelectProps>;
    catalog: readonly GenerationModelCatalogItem[];
    ensureLoaded(): void;
    error: string | null;
    getBaseColorPalette(base: string): string;
    getBaseLabel(base: string): string;
    status: 'error' | 'idle' | 'loaded' | 'loading';
  };
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    reportError(error: { area: string; message: string; namespace: 'generation'; projectId?: string }): void;
  };
  project: {
    activeProjectId: string;
    generateValues: Record<string, unknown>;
    invocationSourceId: string;
    showPromptSyntaxHighlighting: boolean;
  };
  promptHistory: {
    items: readonly PromptHistoryItem[];
    clear(): void;
    remove(prompt: PromptHistoryItem): void;
  };
  settings: {
    patchGenerateSettings(values: Partial<GenerateSettings>, projectId?: string): void;
  };
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
  const { ModelSelect } = useGenerationUi().models;
  return <ModelSelect {...props} />;
};
