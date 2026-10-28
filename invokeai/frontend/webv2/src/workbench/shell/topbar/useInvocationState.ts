import type { DynamicPromptsExpansion } from '@features/generation/react';
import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationRoute, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { useDynamicPrompts } from '@features/generation/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import {
  getEffectivePrompts,
  normalizeGenerateSettings,
  sanitizeBatchCount,
  sanitizeDynamicPromptsConfig,
} from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { isUpscaleBatchCountFixed } from '@features/upscale';
import { getInvocationTemplatesSnapshot, subscribeInvocationTemplates } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { submitActiveInvocation } from '@workbench/activeInvocationSubmission';
import { describeControlLayerIssue } from '@workbench/controlLayerChecks';
import { getPlacedWidgetTypeIds, getVisibleWidgetTypeIds, graphWidgetSources } from '@workbench/graphWidgets';
import {
  createInvocationRouteInputSelector,
  isInvocationRouteValid,
  localizeInvocationValidationReason,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import { useIsInvocationPreparing } from '@workbench/invocationPreparation';
import { getWorkbenchPreferences, subscribeWorkbenchPreferences } from '@workbench/settings/store';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectInvocationRouteInput = createInvocationRouteInputSelector();

/**
 * Include capability, template and remembered control-model snapshots in route selection so each change
 * independently re-evaluates its readiness gate.
 */
const subscribeRouteSources = (listener: () => void): (() => void) => {
  const unsubscribeCapabilities = subscribeArchitectureCapabilities(listener);
  const unsubscribeTemplates = subscribeInvocationTemplates(listener);
  const unsubscribePreferences = subscribeWorkbenchPreferences(listener);

  return () => {
    unsubscribeCapabilities();
    unsubscribeTemplates();
    unsubscribePreferences();
  };
};

let routeSources: {
  capabilities: ReturnType<typeof getArchitectureCapabilitiesSnapshot>;
  controlModelMemory: ReturnType<typeof getWorkbenchPreferences>['controlModelMemory'];
  templates: ReturnType<typeof getInvocationTemplatesSnapshot>;
} | null = null;

/** Referentially stable until one of the stores publishes a new snapshot. */
const getRouteSourcesSnapshot = () => {
  const capabilities = getArchitectureCapabilitiesSnapshot();
  const controlModelMemory = getWorkbenchPreferences().controlModelMemory;
  const templates = getInvocationTemplatesSnapshot();

  if (
    routeSources?.capabilities !== capabilities ||
    routeSources.controlModelMemory !== controlModelMemory ||
    routeSources.templates !== templates
  ) {
    routeSources = { capabilities, controlModelMemory, templates };
  }

  return routeSources;
};

const areTypeIdSetsEqual = (left: ReadonlySet<WidgetTypeId>, right: ReadonlySet<WidgetTypeId>): boolean =>
  left.size === right.size && [...left].every((typeId) => right.has(typeId));

const readDynamicPromptsConfig = (values: Record<string, unknown>) => ({
  combinatorial: values.dynamicPromptsCombinatorial,
  maxPrompts: values.dynamicPromptsMaxPrompts,
  resample: values.dynamicPromptsResample,
  sampleSeed: values.dynamicPromptsSampleSeed,
  seedBehaviour: values.dynamicPromptsSeedBehaviour,
});

const readEffectivePositivePrompt = (values: Record<string, unknown>): string => {
  const settings = normalizeGenerateSettings(values);

  return settings ? getEffectivePrompts(settings).positivePrompt : '';
};

export const isBatchCountFixed = (sourceId: InvocationRoute['sourceId'], values: Record<string, unknown>): boolean =>
  sourceId === 'upscale' && isUpscaleBatchCountFixed(values);

export const getBatchCount = (sourceId: InvocationRoute['sourceId'], values: Record<string, unknown>): number =>
  isBatchCountFixed(sourceId, values) ? 1 : sanitizeBatchCount(values.batchCount);

/** Why Invoke cannot run, worded for the user; `layerId` names the canvas layer to show for it. */
export interface InvocationIssue {
  message: string;
  layerId: string | null;
}

export interface InvocationState {
  batchCount: number;
  /** Every reason Invoke cannot run right now, in the order the tooltip lists them. */
  issues: InvocationIssue[];
  invocation: InvocationRoute;
  /** Sessions one workflow run produces through batch nodes; absent without batch nodes, null while unresolved. */
  workflowBatchSize?: number | null;
  isPreparing: boolean;
  isValid: boolean;
  sources: GraphWidgetSource[];
  visibleTypeIds: ReadonlySet<WidgetTypeId>;
  placedTypeIds: ReadonlySet<WidgetTypeId>;
  promptExpansion: DynamicPromptsExpansion;
  sourceValues: Record<string, unknown>;
  invoke: (options?: InvokeOptions) => Promise<void>;
}

export interface InvokeOptions {
  destinationOverride?: ResultDestination;
  /** Queue ahead of the user's other pending work, as Invoke Front does. */
  prepend?: boolean;
}

export const useInvocationState = (): InvocationState => {
  const { t } = useTranslation();
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const visibleTypeIds = useActiveProjectSelector(getVisibleWidgetTypeIds, areTypeIdSetsEqual);
  const placedTypeIds = useActiveProjectSelector(getPlacedWidgetTypeIds, areTypeIdSetsEqual);
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const { invocation } = routeInput;
  const isPreparing = useIsInvocationPreparing(routeInput.projectId);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  // Resolve inside subscribed selectors: bare imperative reads can remain compiler-memoized after capabilities or
  // templates arrive.
  const resolvedRoute = useExternalStoreSelector(
    subscribeRouteSources,
    getRouteSourcesSnapshot,
    useCallback(
      () => resolveInvocationRouteInput(routeInput, 'global', invocation, availabilityModels),
      [availabilityModels, invocation, routeInput]
    )
  );
  const isConnected = backendConnectionStatus === 'connected';
  const sourceValues =
    invocation.sourceId === 'upscale'
      ? routeInput.upscaleValues
      : invocation.sourceId === 'video'
        ? routeInput.videoValues
        : invocation.sourceId === 'workflow'
          ? routeInput.workflowValues
          : routeInput.generateValues;

  const promptExpansion = useDynamicPrompts(
    readEffectivePositivePrompt(routeInput.generateValues),
    invocation.sourceId === 'generate' || invocation.sourceId === 'canvas'
      ? sanitizeDynamicPromptsConfig(readDynamicPromptsConfig(routeInput.generateValues))
      : null
  );
  // Block unexpandable prompts rather than submitting literal dynamic syntax; hotkey submission enforces the same
  // rule.
  const expansionReason = promptExpansion.isError
    ? 'The prompt could not be expanded.'
    : (promptExpansion.error ?? null);

  const issues = useMemo(
    (): InvocationIssue[] => [
      ...(isConnected ? [] : [{ layerId: null, message: 'The backend is disconnected.' }]),
      ...(expansionReason === null ? [] : [{ layerId: null, message: expansionReason }]),
      ...resolvedRoute.validationReasons.map((reason) => ({
        layerId: typeof reason === 'object' && 'controlLayerIssue' in reason ? reason.controlLayerIssue.layerId : null,
        message: localizeInvocationValidationReason(reason, t),
      })),
    ],
    [expansionReason, isConnected, resolvedRoute.validationReasons, t]
  );
  const isValid = isInvocationRouteValid(resolvedRoute) && isConnected && expansionReason === null;

  const invoke = useCallback(
    ({ destinationOverride, prepend }: InvokeOptions = {}) =>
      submitActiveInvocation({
        commands,
        destinationOverride,
        prepend,
        formatControlLayerError: (rejection) => describeControlLayerIssue(t, rejection),
        formatUpscaleMetadataFailure: (name) => t('widgets.gallery.itemActions.upscale.metadataFailed', { name }),
        getModels: () => availabilityModels,
        queries,
      }),
    [availabilityModels, commands, queries, t]
  );

  return {
    batchCount: getBatchCount(invocation.sourceId, sourceValues),
    invocation,
    invoke,
    issues,
    isPreparing,
    isValid,
    placedTypeIds,
    promptExpansion,
    sourceValues,
    sources: graphWidgetSources,
    visibleTypeIds,
    ...(resolvedRoute.workflowBatch ? { workflowBatchSize: resolvedRoute.workflowBatch.size } : {}),
  };
};
