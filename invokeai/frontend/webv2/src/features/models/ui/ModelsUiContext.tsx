import type { ModelConfig, ModelTaxonomyType } from '@features/models/core/types';
import type { CalibrationGraphCompiler } from '@features/models/core/workingMemory';
import type { ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useNavigate, useRouter } from '@tanstack/react-router';
import { createContext, use, useCallback } from 'react';

export interface SchedulerChoice {
  label: string;
  value: string;
}

/** Generate's scheduler rules for one main model, so its defaults offer exactly what Generate runs. */
export interface ModelSchedulerPolicy {
  /** False where the base's graph ignores the scheduler; Generate hides the control there. */
  appliesToGraph: boolean;
  /** The scheduler Generate starts this base on. */
  defaultValue: string;
  /** Generate's choices for the base, led by `currentValue` when it is outside them. */
  options: readonly SchedulerChoice[];
}

/** The main model's architecture; Generate's scheduler rules key on these alone. */
export type SchedulerPolicyModel = Pick<ModelConfig, 'base' | 'format' | 'variant'>;

export type SchedulerPolicyReader = (model: SchedulerPolicyModel, currentValue: string | null) => ModelSchedulerPolicy;

/** What Generate's component rules read of the main model: its architecture, and the bundle a self-contained one holds. */
export type DefaultComponentSlotModel = Pick<ModelConfig, 'base' | 'format' | 'key' | 'name' | 'submodels' | 'variant'>;

/** One component slot of a main model's family that the model can store a default for. */
export interface DefaultComponentSlotPolicy {
  /** The slot's name in `default_settings.components`. */
  persistenceKey: string;
  label: string;
  /** Generate's help for the slot. */
  helpText?: string;
  modelTypes: readonly ModelTaxonomyType[];
  /** Whether `candidate` may fill the slot, beside the other default components stored. */
  filter: (candidate: ModelConfig) => boolean;
}

/**
 * The slots Generate offers `model`, given the default components stored so far (a slot can depend on another: the
 * vision tower appears only beside a GGUF Qwen3-VL encoder). `catalog` resolves the stored model keys; the VAE has
 * its own default and is not listed.
 */
export type DefaultComponentSlotReader = (
  model: DefaultComponentSlotModel,
  components: Readonly<Record<string, string>>,
  catalog: readonly ModelConfig[]
) => readonly DefaultComponentSlotPolicy[];

/**
 * Generate's rules for a main model's defaults (the scheduler and the component slots): `loading` until the rules and
 * the architecture table they read arrive, `unavailable` when either failed, and `ready` with readers that are
 * replaced whenever the table changes.
 */
export type ModelDefaultsRulesState =
  | { status: 'loading' }
  | { status: 'unavailable' }
  | {
      /** Whether Generate accepts `candidate` as the VAE of `model`, which may be of another base (MOSS-Image takes FLUX.2's). */
      acceptsDefaultVae: (model: DefaultComponentSlotModel, candidate: ModelConfig) => boolean;
      read: SchedulerPolicyReader;
      readComponentSlots: DefaultComponentSlotReader;
      status: 'ready';
    };

/** Generate's rules for main-model defaults, loaded on first use so the Model Manager carries no Generation code at startup. */
export interface ModelDefaultsRulesSource {
  getSnapshot: () => ModelDefaultsRulesState;
  /** Starts loading the rules and the table, or retries after a failure; joins a load already running. */
  load: () => void;
  subscribe: (listener: () => void) => () => void;
}

const DEFAULTS_RULES_UNAVAILABLE: ModelDefaultsRulesState = { status: 'unavailable' };
/** For surfaces composed without Generation, such as isolated tests: the rules never arrive. */
export const NO_DEFAULTS_RULES: ModelDefaultsRulesSource = {
  getSnapshot: () => DEFAULTS_RULES_UNAVAILABLE,
  load: () => undefined,
  subscribe: () => () => undefined,
};

/** This UI port preserves dependency direction: Models cannot import Workbench or Generation. */
export interface ModelsUiAdapter {
  /** Whether this session may open the Model Manager; its route redirects everyone else away. */
  canManageModels: boolean;
  enableModelDescriptions: boolean;
  /** Add Models' "Install in place" choice, an account preference. */
  installInPlace: boolean;
  isProjectActive: (projectId: string) => boolean;
  managerProjectId: string | null;
  defaultsRules: ModelDefaultsRulesSource;
  setInstallInPlace: (installInPlace: boolean) => void;
}

const DEFAULT_MODELS_UI_ADAPTER: ModelsUiAdapter = {
  canManageModels: false,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => false,
  managerProjectId: null,
  defaultsRules: NO_DEFAULTS_RULES,
  setInstallInPlace: () => undefined,
};

const ModelsUiContext = createContext<ModelsUiAdapter>(DEFAULT_MODELS_UI_ADAPTER);

export const ModelsUiProvider = ({ adapter, children }: { adapter: ModelsUiAdapter; children: ReactNode }) => (
  <ModelsUiContext value={adapter}>{children}</ModelsUiContext>
);

export const useModelsUi = (): ModelsUiAdapter => use(ModelsUiContext);

const CalibrationGraphCompilerContext = createContext<CalibrationGraphCompiler | null>(null);

/** Supplies working-memory calibration with Generate's graph compiler, which App composes: Models cannot import it. */
export const CalibrationGraphCompilerProvider = ({
  children,
  compile,
}: {
  children: ReactNode;
  compile: CalibrationGraphCompiler;
}) => <CalibrationGraphCompilerContext value={compile}>{children}</CalibrationGraphCompilerContext>;

/** Null outside a provider; calibration is then unavailable. */
export const useCalibrationGraphCompiler = (): CalibrationGraphCompiler | null => use(CalibrationGraphCompilerContext);

/**
 * Opens the Model Manager on a model's details, returning to the manager's project like its other entry points; null
 * when this session may not manage models. The manager's UI store loads on first use so editor boot does not carry it.
 */
export const useOpenModelInManager = (): ((modelKey: string) => void) | null => {
  const { canManageModels, isProjectActive, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const router = useRouter();
  const open = useCallback(
    (modelKey: string) => {
      const owner = captureAccountScope();
      const location = router.state.location;
      void import('./uiStore').then(({ openModelDetail }) => {
        if (
          !isAccountScopeCurrent(owner) ||
          router.state.location !== location ||
          (managerProjectId !== null && !isProjectActive(managerProjectId))
        ) {
          return;
        }
        openModelDetail(modelKey);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [isProjectActive, managerProjectId, navigate, router]
  );

  return canManageModels ? open : null;
};

/** Opens Add Models searching the starter catalog for `query`, so the user reviews it before installing. */
export const useOpenAddModelsSearch = (): ((query: string) => void) | null => {
  const { canManageModels, isProjectActive, managerProjectId } = useModelsUi();
  const navigate = useNavigate();
  const router = useRouter();
  const open = useCallback(
    (query: string) => {
      const owner = captureAccountScope();
      const location = router.state.location;
      void import('./uiStore').then(({ requestAddModelsSearch }) => {
        if (
          !isAccountScopeCurrent(owner) ||
          router.state.location !== location ||
          (managerProjectId !== null && !isProjectActive(managerProjectId))
        ) {
          return;
        }
        requestAddModelsSearch(query);
        void navigate({ search: { project: managerProjectId ?? undefined }, to: '/models' });
      });
    },
    [isProjectActive, managerProjectId, navigate, router]
  );

  return canManageModels ? open : null;
};
