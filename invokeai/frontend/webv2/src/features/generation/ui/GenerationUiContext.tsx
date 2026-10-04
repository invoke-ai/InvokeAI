import type { GenerationModelCatalogItem, PromptHistoryItem } from '@features/generation/contracts';
import type { RebalancePreset } from '@features/generation/core/conditioningRebalance';
import type { GenerateSettings } from '@features/generation/core/types';
import type { ReadableExternalStore } from '@platform/state/projectedExternalStore';
import type { ComponentType, ReactNode } from 'react';

import { type EqualityFn, shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
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
  scopeLabel?: string;
  showManagerButton?: boolean;
  size?: 'md' | 'lg' | 'xl';
  value: string | null;
}

export interface GenerationSelectedImage {
  imageName: string;
  imageUrl: string;
  thumbnailUrl: string;
}

/** One executed seed from a recent completed Generate run. */
export interface GenerationSeedHistoryItem {
  seed: number;
  thumbnailUrl: string | null;
}

/** A named Generate settings snapshot; `values` is normalized by the feature on apply. */
export interface GeneratePresetRecord {
  id: string;
  label: string;
  values: Record<string, unknown>;
}

/** This port keeps Generation independent of Workbench. */
export interface GenerationUiAdapter {
  CanvasGenerationSections: ComponentType;
  account: {
    currentUserId: string | null;
    multiuserEnabled: boolean;
  };
  capabilities: {
    /** Bulk import/export of prompt templates; the backend routes are admin-only. */
    canManagePromptTemplates: boolean;
    /** Edit prompts shared with everyone; never another user's private prompt. */
    canManageSharedSystemPrompts: boolean;
  };
  gallery: {
    /** Raise Gallery/Preview and locate the image's board, page, and cell. */
    findImage(imageName: string): void;
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
    getImageUrl(key: string): string;
    /** Absent when this session may not manage models. */
    openInModelManager?: (key: string) => void;
    /** Apply the optional model-type filter when opening Add Models. */
    openManager(options?: { modelType?: string }): void;
    status: 'error' | 'idle' | 'loaded' | 'loading';
  };
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    reportError(error: { area: string; message: string; namespace: 'generation'; projectId?: string }): void;
  };
  /** The active project's stored Generate values; a store so edits do not re-render every adapter consumer. */
  generateValues: ReadableExternalStore<Record<string, unknown>>;
  project: {
    activeProjectId: string;
    invocationSourceId: string;
    showPromptSyntaxHighlighting: boolean;
  };
  promptHistory: {
    items: readonly PromptHistoryItem[];
    clear(): void;
    remove(prompt: PromptHistoryItem): void;
  };
  presets: {
    /** User-saved Generate settings snapshots ("recipes"), in save order. */
    presets: readonly GeneratePresetRecord[];
    /** Adds a snapshot under a new id and returns it. */
    save(label: string, values: Record<string, unknown>): GeneratePresetRecord;
    rename(presetId: string, label: string): void;
    remove(presetId: string): void;
  };
  queueInsights: {
    /** Executed seeds of recent completed Generate runs for this project, newest first. */
    seedHistory: readonly GenerationSeedHistoryItem[];
    /** Mean seconds per completed recent Generate run; null with no history to ground it. */
    secondsPerRun: number | null;
  };
  rebalancePresets: {
    /** User-saved conditioning rebalance curves; built-ins are not included. */
    presets: readonly RebalancePreset[];
    /** Adds a curve under a new id and returns it. */
    save(label: string, weights: string, multiplier: number): RebalancePreset;
    rename(presetId: string, label: string): void;
    remove(presetId: string): void;
  };
  sectionPreferences: {
    /** Persisted per-user open/closed overrides for panel sections; absent = section default. */
    sectionsOpen: Readonly<Record<string, boolean>>;
    setSectionOpen(sectionId: string, open: boolean): void;
  };
  settings: {
    patchGenerateSettings(values: Partial<GenerateSettings>, projectId?: string, origin?: 'user' | 'system'): void;
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

const selectAllValues = (values: Record<string, unknown>) => values;

export function useGenerateValues(): Record<string, unknown>;
export function useGenerateValues<Selected>(
  selector: (values: Record<string, unknown>) => Selected,
  isEqual?: EqualityFn<Selected>
): Selected;
export function useGenerateValues<Selected>(
  selector: (values: Record<string, unknown>) => Selected = selectAllValues as never,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected {
  const { generateValues } = useGenerationUi();
  return useExternalStoreSelector(generateValues.subscribe, generateValues.getSnapshot, selector, isEqual);
}

export const GenerationModelSelect = (props: GenerationModelSelectProps) => {
  const { ModelSelect } = useGenerationUi().models;
  return <ModelSelect {...props} />;
};
