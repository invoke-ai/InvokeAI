import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasExportCapability, CanvasLayerCapability } from '@workbench/canvas-engine/api';
import type { ProjectGraphState } from '@workbench/workflows/types';
import type { FormEvent } from 'react';

import { chakra, createListCollection, Dialog, Portal, Stack, Text } from '@chakra-ui/react';
import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph } from '@workbench/canvas-operations/backend/utilityQueue';
import { Button, CloseButton, Field, Select } from '@workbench/components/ui';
import { getGalleryImageByName, makeImageDurable, saveImageToGallery } from '@workbench/gallery/api';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import {
  buildLayerWorkflowGraph,
  getDefaultLayerWorkflowSelection,
  getLayerWorkflowInputs,
  getLayerWorkflowOutputs,
  getRunnableLayerWorkflowInputs,
  reconcileLayerWorkflowSelection,
  type GetRunnableLayerWorkflowInputs,
  type LayerWorkflowSelection,
  type WorkflowImageBinding,
} from '@workbench/workflows/layerWorkflow';
import { useInvocationTemplatesSnapshot, type InvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { createLayerActionSession } from './layerActionSession';
import {
  runLayerWorkflow,
  type LayerWorkflowDestination,
  type LayerWorkflowFailureStage,
  type RunLayerWorkflowResult,
} from './runLayerWorkflow';

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;
const EMPTY_BINDINGS: readonly WorkflowImageBinding[] = [];

const bindingKey = (binding: WorkflowImageBinding): string => JSON.stringify([binding.nodeId, binding.fieldName]);
const messageOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export interface LayerWorkflowAvailability {
  canRunWorkflow: boolean;
  document: ProjectGraphState;
  getRunnableInputs(output: WorkflowImageBinding): readonly WorkflowImageBinding[];
  hasWorkflowBindings: boolean;
  inputs: readonly WorkflowImageBinding[];
  outputs: readonly WorkflowImageBinding[];
  templatesSnapshot: InvocationTemplatesSnapshot;
}

/**
 * Every layer row observes the same immutable graph/template objects. Cache the
 * expensive injected-graph readiness probes by those identities so a long
 * layer list does the work once, while each row still gets live eligibility.
 */
const availabilityCache = new WeakMap<
  ProjectGraphState,
  WeakMap<InvocationTemplatesSnapshot, LayerWorkflowAvailability>
>();

const getLayerWorkflowAvailability = (
  document: ProjectGraphState,
  templatesSnapshot: InvocationTemplatesSnapshot
): LayerWorkflowAvailability => {
  let byTemplates = availabilityCache.get(document);

  if (!byTemplates) {
    byTemplates = new WeakMap();
    availabilityCache.set(document, byTemplates);
  }

  const cached = byTemplates.get(templatesSnapshot);

  if (cached) {
    return cached;
  }

  const inputs =
    templatesSnapshot.status === 'loaded'
      ? getLayerWorkflowInputs(document, templatesSnapshot.templates)
      : EMPTY_BINDINGS;
  const outputs =
    templatesSnapshot.status === 'loaded'
      ? getLayerWorkflowOutputs(document, templatesSnapshot.templates)
      : EMPTY_BINDINGS;
  const runnableByOutput = new Map<string, readonly WorkflowImageBinding[]>();

  for (const output of outputs) {
    runnableByOutput.set(bindingKey(output), getRunnableLayerWorkflowInputs(document, templatesSnapshot, output));
  }

  const availability: LayerWorkflowAvailability = {
    canRunWorkflow: [...runnableByOutput.values()].some((runnableInputs) => runnableInputs.length > 0),
    document,
    getRunnableInputs: (output) => runnableByOutput.get(bindingKey(output)) ?? EMPTY_BINDINGS,
    hasWorkflowBindings: inputs.length > 0 && outputs.length > 0,
    inputs,
    outputs,
    templatesSnapshot,
  };

  byTemplates.set(templatesSnapshot, availability);
  return availability;
};

export const useLayerWorkflowAvailability = (): LayerWorkflowAvailability => {
  const document = useActiveProjectSelector((project) => project.projectGraph, Object.is);
  const templatesSnapshot = useInvocationTemplatesSnapshot();

  return getLayerWorkflowAvailability(document, templatesSnapshot);
};

interface RunLayerWorkflowDialogProps {
  availability: LayerWorkflowAvailability;
  engine: { readonly exports: CanvasExportCapability; readonly layers: CanvasLayerCapability } | null;
  isOpen: boolean;
  layerId: string;
  onClose(): void;
}

interface SelectionState {
  availability: LayerWorkflowAvailability;
  selection: LayerWorkflowSelection;
}

const createDefaultSelection = (availability: LayerWorkflowAvailability): LayerWorkflowSelection =>
  getDefaultLayerWorkflowSelection(availability.outputs, availability.getRunnableInputs);

export const reconcileLayerWorkflowDialogSelection = (
  current: LayerWorkflowSelection,
  outputs: readonly WorkflowImageBinding[],
  getRunnableInputs: GetRunnableLayerWorkflowInputs
): LayerWorkflowSelection => {
  const selectedOutput = current.output;
  const currentOutput = selectedOutput
    ? outputs.find((output) => bindingKey(output) === bindingKey(selectedOutput))
    : undefined;

  if (currentOutput) {
    return reconcileLayerWorkflowSelection(current, currentOutput, getRunnableInputs(currentOutput));
  }

  return {
    ...getDefaultLayerWorkflowSelection(outputs, getRunnableInputs),
    destination: current.destination,
  };
};

const SUCCESS_KEYS: Record<LayerWorkflowDestination, string> = {
  'copy-raster': 'widgets.layers.runWorkflow.copySuccess',
  gallery: 'widgets.layers.runWorkflow.gallerySuccess',
  replace: 'widgets.layers.runWorkflow.replaceSuccess',
  staging: 'widgets.layers.runWorkflow.stagingSuccess',
};

const FAILURE_KEYS: Record<Exclude<RunLayerWorkflowResult['status'], 'completed' | 'failed'>, string> = {
  aborted: 'widgets.layers.runWorkflow.aborted',
  busy: 'widgets.layers.runWorkflow.busy',
  disabled: 'widgets.layers.runWorkflow.disabled',
  empty: 'widgets.layers.runWorkflow.empty',
  locked: 'widgets.layers.runWorkflow.locked',
  missing: 'widgets.layers.runWorkflow.missing',
  'not-ready': 'widgets.layers.runWorkflow.notReady',
  'over-budget': 'widgets.layers.runWorkflow.notReady',
  stale: 'widgets.layers.runWorkflow.stale',
  unsupported: 'widgets.layers.runWorkflow.unsupported',
};

export const getLayerWorkflowFailureKey = (stage: LayerWorkflowFailureStage): string => {
  switch (stage) {
    case 'graph':
      return 'widgets.layers.runWorkflow.graphFailure';
    case 'hydrate':
      return 'widgets.layers.runWorkflow.hydrationFailure';
    case 'durability':
      return 'widgets.layers.runWorkflow.durabilityFailure';
    default:
      return 'widgets.layers.runWorkflow.failed';
  }
};

export const RunLayerWorkflowDialog = ({
  availability,
  engine,
  isOpen,
  layerId,
  onClose,
}: RunLayerWorkflowDialogProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const dispatch = useWorkbenchDispatch();
  const projectId = useActiveProjectSelector((project) => project.id);
  const [session] = useState(createLayerActionSession);
  const [selectionState, setSelectionState] = useState<SelectionState>(() => ({
    availability,
    selection: createDefaultSelection(availability),
  }));
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  let selection = selectionState.selection;

  if (selectionState.availability !== availability) {
    selection = reconcileLayerWorkflowDialogSelection(selection, availability.outputs, availability.getRunnableInputs);
    setSelectionState({ availability, selection });
  }

  const runnableInputs = selection.output ? availability.getRunnableInputs(selection.output) : EMPTY_BINDINGS;
  const inputCollection = useMemo(
    () =>
      createListCollection({
        items: runnableInputs.map((binding) => ({ label: binding.label, value: bindingKey(binding) })),
      }),
    [runnableInputs]
  );
  const outputCollection = useMemo(
    () =>
      createListCollection({
        items: availability.outputs.map((binding) => ({ label: binding.label, value: bindingKey(binding) })),
      }),
    [availability.outputs]
  );
  const destinationCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: LayerWorkflowDestination }>({
        items: [
          { label: t('widgets.layers.runWorkflow.gallery'), value: 'gallery' },
          { label: t('widgets.layers.runWorkflow.staging'), value: 'staging' },
          { label: t('widgets.layers.runWorkflow.replace'), value: 'replace' },
          { label: t('widgets.layers.runWorkflow.copyRaster'), value: 'copy-raster' },
        ],
      }),
    [t]
  );
  const inputValue = useMemo(() => (selection.input ? [bindingKey(selection.input)] : []), [selection.input]);
  const outputValue = useMemo(() => (selection.output ? [bindingKey(selection.output)] : []), [selection.output]);
  const destinationValue = useMemo(() => [selection.destination], [selection.destination]);

  const close = useCallback(() => {
    session.cancel();
    onClose();
  }, [onClose, session]);

  const contentRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node) {
        return;
      }
      return () => {
        session.cancel();
      };
    },
    [session]
  );

  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        close();
      }
    },
    [close]
  );

  const handleInputChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const input = runnableInputs.find((binding) => bindingKey(binding) === value[0]);

      if (!input) {
        return;
      }

      setSelectionState((current) => ({
        availability,
        selection: { ...current.selection, input },
      }));
      setError(null);
    },
    [availability, runnableInputs]
  );

  const handleOutputChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const output = availability.outputs.find((binding) => bindingKey(binding) === value[0]);

      if (!output) {
        return;
      }

      setSelectionState((current) => ({
        availability,
        selection: reconcileLayerWorkflowSelection(current.selection, output, availability.getRunnableInputs(output)),
      }));
      setError(null);
    },
    [availability]
  );

  const handleDestinationChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const destination = value[0] as LayerWorkflowDestination | undefined;

      if (!destination || !destinationCollection.items.some((item) => item.value === destination)) {
        return;
      }

      setSelectionState((current) => ({
        availability,
        selection: { ...current.selection, destination },
      }));
      setError(null);
    },
    [availability, destinationCollection.items]
  );

  const runContext = useMemo(
    () => ({
      availability,
      close,
      dispatch,
      engine,
      layerId,
      notify,
      projectId,
      selection,
      session,
      t,
    }),
    [availability, close, dispatch, engine, layerId, notify, projectId, selection, session, t]
  );

  const run = useCallback(async (): Promise<void> => {
    const { availability, close, dispatch, engine, layerId, notify, projectId, selection, session, t } = runContext;
    const { input, output } = selection;

    if (!engine || !input || !output) {
      return;
    }

    const request = session.begin();

    if (!request) {
      return;
    }

    const destination = selection.destination;
    setError(null);
    setIsRunning(true);

    try {
      const executorDeps = engine.exports.getCompositeExecutorDeps();
      const result = await runLayerWorkflow({
        deps: {
          appendStaging: (targetProjectId, candidate) =>
            dispatch({ candidate, projectId: targetProjectId, type: 'appendCanvasStagingCandidate' }),
          buildGraph: buildLayerWorkflowGraph,
          commitGenerated: (options) => engine.layers.commitGeneratedImageResult(options),
          createRequestId: () => crypto.randomUUID(),
          exportLayer: (targetLayerId) => engine.exports.exportBakedLayerBlob(targetLayerId, { includeDisabled: true }),
          getImage: getGalleryImageByName,
          isGuardCurrent: (guard) => engine.exports.isLayerExportGuardCurrent(guard),
          makeDurable: makeImageDurable,
          runGraph: (options) => runUtilityGraph({ ...options, hub: socketHub }),
          saveToGallery: saveImageToGallery,
          touchGallery: (targetProjectId) =>
            dispatch({ projectId: targetProjectId, type: 'touchGalleryImagesRefresh' }),
          uploadIntermediate: async (blob, signal) => {
            if (signal?.aborted) {
              throw new DOMException('Layer workflow aborted', 'AbortError');
            }
            const uploaded = await executorDeps.uploadImage(blob);
            if (signal?.aborted) {
              throw new DOMException('Layer workflow aborted', 'AbortError');
            }
            return { imageName: uploaded.imageName };
          },
        },
        destination,
        document: availability.document,
        input,
        layerId,
        output,
        projectId,
        signal: request.signal,
        templatesSnapshot: availability.templatesSnapshot,
      });

      if (!session.isCurrent(request.token)) {
        return;
      }

      if (result.status === 'completed') {
        notify.success(t(SUCCESS_KEYS[destination]));
        close();
        return;
      }

      if (result.status === 'failed') {
        setError(t(getLayerWorkflowFailureKey(result.stage), { message: result.message }));
        return;
      }

      setError(t(FAILURE_KEYS[result.status]));
    } catch (error) {
      if (session.isCurrent(request.token)) {
        setError(t('widgets.layers.runWorkflow.failed', { message: messageOf(error) }));
      }
    } finally {
      if (session.isCurrent(request.token)) {
        setIsRunning(false);
      }
      session.finish(request.token);
    }
  }, [runContext]);

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void run();
    },
    [run]
  );

  const readinessMessage = !availability.hasWorkflowBindings
    ? t('widgets.layers.runWorkflow.noBindings')
    : !selection.input || !selection.output
      ? t('widgets.layers.runWorkflow.notReady')
      : null;
  const canRun = engine !== null && !isRunning && readinessMessage === null;

  return (
    <Dialog.Root lazyMount open={isOpen} placement="center" size="sm" unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content ref={contentRef} bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  {t('widgets.layers.runWorkflow.title')}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="4">
                  <Field id="layer-workflow-output" label={t('widgets.layers.runWorkflow.output')}>
                    <Select
                      aria-label={t('widgets.layers.runWorkflow.output')}
                      collection={outputCollection}
                      disabled={isRunning || availability.outputs.length === 0}
                      positioning={SELECT_POSITIONING}
                      size="sm"
                      value={outputValue}
                      onValueChange={handleOutputChange}
                    />
                  </Field>
                  <Field id="layer-workflow-input" label={t('widgets.layers.runWorkflow.input')}>
                    <Select
                      aria-label={t('widgets.layers.runWorkflow.input')}
                      collection={inputCollection}
                      disabled={isRunning || runnableInputs.length === 0}
                      positioning={SELECT_POSITIONING}
                      size="sm"
                      value={inputValue}
                      onValueChange={handleInputChange}
                    />
                  </Field>
                  <Field id="layer-workflow-destination" label={t('widgets.layers.runWorkflow.destination')}>
                    <Select
                      aria-label={t('widgets.layers.runWorkflow.destination')}
                      collection={destinationCollection}
                      disabled={isRunning}
                      positioning={SELECT_POSITIONING}
                      size="sm"
                      value={destinationValue}
                      onValueChange={handleDestinationChange}
                    />
                  </Field>
                  {readinessMessage ? (
                    <Text color="fg.error" fontSize="xs" role="alert">
                      {readinessMessage}
                    </Text>
                  ) : null}
                  {error ? (
                    <Text color="fg.error" fontSize="xs" role="alert">
                      {error}
                    </Text>
                  ) : null}
                  {isRunning ? (
                    <Text color="fg.muted" fontSize="xs" role="status">
                      {t('widgets.layers.runWorkflow.running')}
                    </Text>
                  ) : null}
                </Stack>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" type="button" variant="ghost" onClick={close}>
                  {t('widgets.layers.runWorkflow.cancel')}
                </Button>
                <Button disabled={!canRun} loading={isRunning} size="xs" type="submit" variant="solid">
                  {t('widgets.layers.runWorkflow.run')}
                </Button>
              </Dialog.Footer>
            </chakra.form>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('widgets.layers.runWorkflow.cancel')} color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
