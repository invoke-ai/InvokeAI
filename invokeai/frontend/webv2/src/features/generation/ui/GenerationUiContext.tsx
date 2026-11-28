import type { GenerationModelCatalogItem, PromptHistoryItem } from '@features/generation/contracts';
import type { AspectRatioPresetsPort } from '@features/generation/core/aspectRatioPresets';
import type { RebalancePreset } from '@features/generation/core/conditioningRebalance';
import type { GenerateSettings } from '@features/generation/core/types';
import type { ReadableExternalStore } from '@platform/state/projectedExternalStore';
import type { ComponentType, ReactNode } from 'react';

import { type EqualityFn, shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { createContext, use } from 'react';

export interface GenerationModelSelectProps {
  className?: string;
  disabled?: boolean;
  emptyLabel?: string;
  excludeKeys?: ReadonlySet<string>;
  filter?: (model: GenerationModelCatalogItem) => boolean;
  id?: string;
  invalid?: boolean;
  isClearable?: boolean;
  modelTypes: string[];
  onChange: (model: GenerationModelCatalogItem | null) => void;
  placeholder?: string;
  relatedTo?: string | null;
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

export interface GenerationSeedHistoryItem {
  seed: number;
  thumbnailUrl: string | null;
}

export interface GenerationQueueInsights {
  /** Executed seeds of recent completed Generate runs for this project, newest first. */
  seedHistory: readonly GenerationSeedHistoryItem[];
  /** Mean seconds per completed recent Generate run; null with no history to ground it. */
  secondsPerRun: number | null;
}

/** A named Generate settings snapshot; `values` is normalized by the feature on apply. */
export interface GeneratePresetRecord {
  id: string;
  label: string;
  values: Record<string, unknown>;
}

export interface CanvasRenderSize {
  controls: ReactNode;
  /** The result is generated at this size, then resized to the frame; null until a model is selected. */
  size: { height: number; width: number } | null;
}

export interface CanvasRenderSizeProps {
  children: (renderSize: CanvasRenderSize) => ReactNode;
  /** The frame as the Size section shows it, including a resize that has not been committed yet. */
  frame: { height: number; width: number };
}

export interface CanvasDenoisingStrength {
  summary: ReactNode;
  field: ReactNode;
}

export interface CanvasDenoisingStrengthProps {
  children: (strength: CanvasDenoisingStrength) => ReactNode;
}

/** This port keeps Generation independent of Workbench. */
export interface GenerationUiAdapter {
  /** Wraps the Render section in canvas mode, which adds denoising strength to it. */
  CanvasDenoisingStrength: ComponentType<CanvasDenoisingStrengthProps>;
  CanvasGenerationSections: ComponentType;
  /** Wraps the Size section in canvas mode, where the frame and the render size can differ. */
  CanvasRenderSize: ComponentType<CanvasRenderSizeProps>;
  account: {
    currentUserId: string | null;
    multiuserEnabled: boolean;
  };
  /** The account's saved aspect ratios, shared with the canvas Frame tool. */
  aspectRatioPresets: AspectRatioPresetsPort;
  capabilities: {
    /** Bulk import/export of prompt templates; the backend routes are admin-only. */
    canManagePromptTemplates: boolean;
    /** Edit prompts shared with everyone; never another user's private prompt. */
    canManageSharedSystemPrompts: boolean;
  };
  gallery: {
    findImage(imageName: string): void;
    selectedImage: GenerationSelectedImage | null;
    touchImages(): void;
    /** Resolved at upload: the auto-add board while addable, else the project inbox; `none` is Uncategorized. */
    getUploadBoardId(): Promise<string>;
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
    openManager(options?: { modelType?: string }): void;
    /** The catalog as it is now, for an Undo that runs long after the render that offered it. */
    readCatalog(): readonly GenerationModelCatalogItem[];
    /** Reads the catalog again, even after a read that half failed; settles when the read does. */
    refresh(): Promise<void>;
    status: 'error' | 'idle' | 'loaded' | 'loading';
  };
  notifications: {
    error(title: string, message?: string): void;
    info(title: string, message?: string): void;
    /** A toast choice that stays until picked or closed, closing with the account; `onClick` rechecks its target. */
    offer(
      title: string,
      notice: { actions: readonly { label: string; onClick: () => void }[]; message?: string }
    ): void;
    reportError(error: { area: string; message: string; namespace: 'generation'; projectId?: string }): void;
    /** `onUndo` must capture the ids it reverts; with `projectId`, Undo is withdrawn once that project closes. */
    undoable(
      title: string,
      notice: { message?: string; onUndo: () => void; projectId?: string; supersedes?: string }
    ): void;
  };
  /** The active project's stored Generate values; a store so edits do not re-render every adapter consumer. */
  generateValues: ReadableExternalStore<Record<string, unknown>>;
  project: {
    activeProjectId: string;
    invocationSourceId: string;
    showPromptSyntaxHighlighting: boolean;
  };
  promptHistory: {
    /** The active project's history. */
    items: readonly PromptHistoryItem[];
    clear(projectId: string): void;
    remove(prompt: PromptHistoryItem): void;
    /** Undoes a clear: puts `items` back into that project's history, behind entries added since. */
    restore(items: readonly PromptHistoryItem[], projectId: string): void;
  };
  presets: {
    /** Whether a new preset keeps the prompts too; the save dialog's last choice, remembered per account. */
    includePrompts: boolean;
    setIncludePrompts(includePrompts: boolean): void;
    presets: readonly GeneratePresetRecord[];
    save(label: string, values: Record<string, unknown>): GeneratePresetRecord;
    rename(presetId: string, label: string): void;
    remove(presetId: string): void;
    /** Undoes a remove: puts `preset` back at `index` (or last), unless a preset with its id is still there. */
    restore(preset: GeneratePresetRecord, index: number): void;
    /** Replaces a preset's snapshot, keeping its id and label. */
    update(presetId: string, values: Record<string, unknown>): void;
  };
  /** Subscribing keeps the queue read observed, so only displaying controls read it (`useGenerationQueueInsights`). */
  queueInsights: ReadableExternalStore<GenerationQueueInsights>;
  rebalancePresets: {
    /** User-saved conditioning rebalance curves; built-ins are not included. */
    presets: readonly RebalancePreset[];
    save(label: string, weights: string, multiplier: number): RebalancePreset;
    rename(presetId: string, label: string): void;
    remove(presetId: string): void;
    /** Undoes a remove: puts `preset` back at `index` among the saved curves (or last), unless its id is still there. */
    restore(preset: RebalancePreset, index: number): void;
  };
  sectionPreferences: {
    /** Persisted per-user open/closed overrides for panel sections; absent = section default. */
    sectionsOpen: Readonly<Record<string, boolean>>;
    /** Records several sections' open states as one preference write (a panel-wide fold or expand). */
    setSectionsOpen(changes: Readonly<Record<string, boolean>>): void;
  };
  settings: {
    patchGenerateSettings(values: Partial<GenerateSettings>, projectId?: string, origin?: 'user' | 'system'): void;
    /** A project's stored Generate values, read when an Undo runs; null once the project is closed. */
    readGenerateValues(projectId: string): Record<string, unknown> | null;
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

export const useGenerationQueueInsights = <Selected,>(
  selector: (insights: GenerationQueueInsights) => Selected,
  isEqual: EqualityFn<Selected> = shallowEqual
): Selected => {
  const { queueInsights } = useGenerationUi();
  return useExternalStoreSelector(queueInsights.subscribe, queueInsights.getSnapshot, selector, isEqual);
};

export const GenerationModelSelect = (props: GenerationModelSelectProps) => {
  const { ModelSelect } = useGenerationUi().models;
  return <ModelSelect {...props} />;
};
