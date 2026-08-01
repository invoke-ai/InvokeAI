import type { DynamicPromptsExpansion } from '@features/generation/react';
import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationRoute, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { useDynamicPrompts } from '@features/generation/react';
import {
  getEffectivePrompts,
  normalizeGenerateSettings,
  sanitizeBatchCount,
  sanitizeDynamicPromptsConfig,
} from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { submitActiveInvocation } from '@workbench/activeInvocationSubmission';
import { getPlacedWidgetTypeIds, getVisibleWidgetTypeIds, graphWidgetSources } from '@workbench/graphWidgets';
import {
  createInvocationRouteInputSelector,
  isInvocationRouteValid,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectInvocationRouteInput = createInvocationRouteInputSelector();
const areTypeIdSetsEqual = (left: ReadonlySet<WidgetTypeId>, right: ReadonlySet<WidgetTypeId>): boolean =>
  left.size === right.size && [...left].every((typeId) => right.has(typeId));

/** Reads the persisted dynamic prompt fields off untyped widget values. */
const readDynamicPromptsConfig = (values: Record<string, unknown>) => ({
  combinatorial: values.dynamicPromptsCombinatorial,
  maxPrompts: values.dynamicPromptsMaxPrompts,
  sampleSeed: values.dynamicPromptsSampleSeed,
  seedBehaviour: values.dynamicPromptsSeedBehaviour,
});

/**
 * The prompt that will actually be expanded and submitted — the authored text
 * wrapped by the active template, since a template can introduce `{a|b}` the
 * authored prompt never had.
 */
const readEffectivePositivePrompt = (values: Record<string, unknown>): string => {
  const settings = normalizeGenerateSettings(values);

  return settings ? getEffectivePrompts(settings).positivePrompt : '';
};

export const getBatchCount = (values: Record<string, unknown>): number => sanitizeBatchCount(values.batchCount);

export interface InvocationState {
  batchCount: number;
  blockingReasons: string[];
  invocation: InvocationRoute;
  isValid: boolean;
  /** Every graph widget — all of them stay selectable in the routing menu. */
  sources: GraphWidgetSource[];
  /** Widget types the layout is showing right now. */
  visibleTypeIds: ReadonlySet<WidgetTypeId>;
  /** Widget types placed in some rail; picking one of these needs no new placement. */
  placedTypeIds: ReadonlySet<WidgetTypeId>;
  promptExpansion: DynamicPromptsExpansion;
  /** The values the batch count and prompt summary belong to (generate or upscale). */
  sourceValues: Record<string, unknown>;
  /** Submits the current route. Pass a destination to override it for this run only. */
  invoke: (destinationOverride?: ResultDestination) => Promise<void>;
}

/**
 * Everything the invocation cluster needs to render and to fire, in one place.
 *
 * The Invoke button, the routing indicator, and the iterations stepper all
 * depend on the same resolved route; splitting the resolution across three
 * components would let them disagree about whether the route is runnable, which
 * is exactly the failure the redesign exists to prevent.
 */
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

  // Project-graph route validation reads the invocation templates imperatively;
  // subscribing here keeps the resolved route live while they load.
  useInvocationTemplatesSelector((snapshot) => snapshot.status);
  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const resolvedRoute = resolveInvocationRouteInput(routeInput, 'global', invocation, availabilityModels);
  const isConnected = backendConnectionStatus === 'connected';
  const sourceValues = invocation.sourceId === 'upscale' ? routeInput.upscaleValues : routeInput.generateValues;

  // Observing the shared expansion cache here keeps the count honest and costs
  // nothing extra: the Generate preview and the submit path use the same key.
  const promptExpansion = useDynamicPrompts(
    readEffectivePositivePrompt(routeInput.generateValues),
    invocation.sourceId === 'generate' || invocation.sourceId === 'canvas'
      ? sanitizeDynamicPromptsConfig(readDynamicPromptsConfig(routeInput.generateValues))
      : null
  );
  // A prompt that will not expand cannot be submitted — the alternative is putting
  // the literal `{a|b}` in front of the model — so this blocks rather than just
  // annotating. `submitResolvedInvocation` enforces the same rule for the hotkey.
  const expansionReason = promptExpansion.isError
    ? 'The prompt could not be expanded.'
    : (promptExpansion.error ?? null);

  // Whether the source widget is *placed* is already part of route validation
  // ("The X widget is not mounted in this project"), so there is no separate
  // openness gate here — a source that is merely not on screen still runs.
  const blockingReasons = useMemo(
    () => [
      ...(isConnected ? [] : ['The backend is disconnected.']),
      ...(expansionReason === null ? [] : [expansionReason]),
      ...resolvedRoute.validationReasons,
    ],
    [expansionReason, isConnected, resolvedRoute.validationReasons]
  );
  const isValid = isInvocationRouteValid(resolvedRoute) && isConnected && expansionReason === null;

  const invoke = useCallback(
    (destinationOverride?: ResultDestination) =>
      submitActiveInvocation({
        commands,
        destinationOverride,
        formatControlLayerError: (code, layerName) =>
          t('widgets.layers.control.invalidLayer', {
            name: layerName,
            reason: t(`widgets.layers.control.validation.${code}`),
          }),
        getModels: () => availabilityModels,
        queries,
      }),
    [availabilityModels, commands, queries, t]
  );

  return {
    batchCount: getBatchCount(sourceValues),
    blockingReasons,
    invocation,
    invoke,
    isValid,
    placedTypeIds,
    promptExpansion,
    sourceValues,
    sources: graphWidgetSources,
    visibleTypeIds,
  };
};
