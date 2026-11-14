import type { DynamicPromptsExpansion } from '@features/generation/react';
import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationRoute, InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { useDynamicPrompts } from '@features/generation/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import {
  describeDynamicPromptsFailure,
  describeGenerationIssue,
  getDynamicPromptsConfig,
  getEffectivePrompts,
  normalizeGenerateSettings,
  sanitizeBatchCount,
  sanitizeDynamicPromptsConfig,
} from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { isUpscaleBatchCountFixed } from '@features/upscale';
import { normalizeVideoWidgetValues } from '@features/video';
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

import { getInvocationIssueTarget, type InvocationIssue } from './invokeIssueTargets';

export type { InvocationIssue, InvocationIssueTarget } from './invokeIssueTargets';

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
  enabled: values.dynamicPromptsEnabled,
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
  /** Hears that the run has started (valid, and holding the project's preparation lease). */
  onStarted?: () => void;
}

/**
 * Invoke's state for the active route; given a `sourceId`, for that source's route instead, its source fixed (a graph
 * preview's Invoke), with the active route's destination.
 */
export const useInvocationState = ({ sourceId }: { sourceId?: InvocationSourceId } = {}): InvocationState => {
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
  const invocation = useMemo(
    () => (sourceId ? { ...routeInput.invocation, sourceId, sourceLocked: true } : routeInput.invocation),
    [routeInput.invocation, sourceId]
  );
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
      () => resolveInvocationRouteInput(routeInput, sourceId ? 'dialog' : 'global', invocation, availabilityModels),
      [availabilityModels, invocation, routeInput, sourceId]
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

  // Video reads its own normalized values, so the count follows what its submission expands.
  const videoValues = invocation.sourceId === 'video' ? normalizeVideoWidgetValues(routeInput.videoValues) : null;
  const promptExpansion = useDynamicPrompts(
    videoValues ? videoValues.positivePrompt : readEffectivePositivePrompt(routeInput.generateValues),
    videoValues
      ? getDynamicPromptsConfig(videoValues)
      : invocation.sourceId === 'generate' || invocation.sourceId === 'canvas'
        ? sanitizeDynamicPromptsConfig(readDynamicPromptsConfig(routeInput.generateValues))
        : null
  );
  // Block unexpandable prompts rather than submitting literal dynamic syntax; hotkey submission enforces the same
  // rule. The prompt field offers the way out: retry, or turn dynamic prompts off and run the prompt as typed.
  const expansionFailure = describeDynamicPromptsFailure(promptExpansion);
  const expansionReason = expansionFailure ? describeGenerationIssue(t, expansionFailure) : null;

  const issues = useMemo(
    (): InvocationIssue[] => [
      ...(isConnected
        ? []
        : [{ message: t('topbar.invoke.serverDisconnected'), target: { kind: 'connection' } } as const]),
      ...(expansionReason === null
        ? []
        : [
            {
              message: expansionReason,
              target: { kind: 'prompt', widgetId: invocation.sourceId === 'video' ? 'video' : 'generate' },
            } as const,
          ]),
      ...resolvedRoute.validationReasons.map((reason) => ({
        message: localizeInvocationValidationReason(reason, t),
        target: getInvocationIssueTarget(reason, invocation.sourceId),
      })),
    ],
    [expansionReason, invocation.sourceId, isConnected, resolvedRoute.validationReasons, t]
  );
  const isValid = isInvocationRouteValid(resolvedRoute) && isConnected && expansionReason === null;

  const invoke = useCallback(
    ({ destinationOverride, onStarted, prepend }: InvokeOptions = {}) =>
      submitActiveInvocation({
        commands,
        destinationOverride,
        onStarted,
        prepend,
        sourceId,
        formatControlLayerError: (rejection) => describeControlLayerIssue(t, rejection),
        formatUpscaleMetadataFailure: (name) => t('widgets.gallery.itemActions.upscale.metadataFailed', { name }),
        getModels: () => availabilityModels,
        queries,
      }),
    [availabilityModels, commands, queries, sourceId, t]
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
