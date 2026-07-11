import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { LayerWorkflowSelection, WorkflowImageBinding } from '@workbench/workflows/layerWorkflow';
import type { ChangeEvent } from 'react';

import { HStack, NativeSelect, Text } from '@chakra-ui/react';
import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph } from '@workbench/canvas-engine/backend/utilityQueue';
import { Button } from '@workbench/components/ui';
import { getGalleryImageByName, makeImageDurable, saveImageToGallery } from '@workbench/gallery/api';
import { useNotify } from '@workbench/useNotify';
import {
  createDefaultLayerWorkflowSelection,
  getLayerWorkflowFailureKey,
  layerWorkflowBindingKey,
  reconcileLayerWorkflowOperationSelection,
  useLayerWorkflowAvailability,
  type LayerWorkflowAvailability,
} from '@workbench/widgets/layers/layerWorkflowAvailability';
import {
  runLayerWorkflow,
  type LayerWorkflowDestination,
  type RunLayerWorkflowResult,
} from '@workbench/widgets/layers/runLayerWorkflow';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { buildLayerWorkflowGraph, reconcileLayerWorkflowSelection } from '@workbench/workflows/layerWorkflow';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

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
  stale: 'widgets.layers.runWorkflow.stale',
  unsupported: 'widgets.layers.runWorkflow.unsupported',
};

const messageOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));
const EMPTY_BINDINGS: readonly WorkflowImageBinding[] = [];

interface SelectionState {
  availability: LayerWorkflowAvailability;
  selection: LayerWorkflowSelection;
}

export interface WorkflowActionEligibility {
  canCancel: boolean;
  canEdit: boolean;
  canRun: boolean;
}

export const getWorkflowActionEligibility = ({
  hasInput,
  hasOutput,
  isRunning,
}: {
  hasInput: boolean;
  hasOutput: boolean;
  isRunning: boolean;
}): WorkflowActionEligibility => ({
  canCancel: true,
  canEdit: !isRunning,
  canRun: !isRunning && hasInput && hasOutput,
});

const findBinding = (bindings: readonly WorkflowImageBinding[], value: string): WorkflowImageBinding | undefined =>
  bindings.find((binding) => layerWorkflowBindingKey(binding) === value);

export const WorkflowOptions = ({ engine, layerId }: { engine: CanvasEngine; layerId: string }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const dispatch = useWorkbenchDispatch();
  const projectId = useActiveProjectSelector((project) => project.id);
  const availability = useLayerWorkflowAvailability();
  const [selectionState, setSelectionState] = useState<SelectionState>(() => ({
    availability,
    selection: createDefaultLayerWorkflowSelection(availability),
  }));
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  let selection = selectionState.selection;
  if (selectionState.availability !== availability) {
    selection = reconcileLayerWorkflowOperationSelection(
      selection,
      availability.outputs,
      availability.getRunnableInputs
    );
    setSelectionState({ availability, selection });
  }
  const runnableInputs = selection.output ? availability.getRunnableInputs(selection.output) : EMPTY_BINDINGS;
  const eligibility = getWorkflowActionEligibility({
    hasInput: selection.input !== null,
    hasOutput: selection.output !== null,
    isRunning,
  });

  const updateSelection = useCallback(
    (next: LayerWorkflowSelection) => {
      setSelectionState({ availability, selection: next });
      setError(null);
    },
    [availability]
  );
  const handleOutput = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const output = findBinding(availability.outputs, event.currentTarget.value);
      if (output) {
        updateSelection(reconcileLayerWorkflowSelection(selection, output, availability.getRunnableInputs(output)));
      }
    },
    [availability, selection, updateSelection]
  );
  const handleInput = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const input = findBinding(runnableInputs, event.currentTarget.value);
      if (input) {
        updateSelection({ ...selection, input });
      }
    },
    [runnableInputs, selection, updateSelection]
  );
  const handleDestination = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      updateSelection({ ...selection, destination: event.currentTarget.value as LayerWorkflowDestination });
    },
    [selection, updateSelection]
  );
  const cancel = useCallback(() => engine.cancelWorkflowOperation(), [engine]);
  const run = useCallback(async () => {
    const { input, output } = selection;
    if (!input || !output || isRunning) {
      return;
    }
    setError(null);
    setIsRunning(true);
    const destination = selection.destination;
    try {
      const operationResult = await engine.runWorkflowOperation(async (signal) => {
        const executorDeps = engine.getCompositeExecutorDeps();
        const result = await runLayerWorkflow({
          deps: {
            appendStaging: (targetProjectId, candidate) =>
              dispatch({ candidate, projectId: targetProjectId, type: 'appendCanvasStagingCandidate' }),
            buildGraph: buildLayerWorkflowGraph,
            commitGenerated: (options) => engine.commitGeneratedImageResult(options),
            createRequestId: () => crypto.randomUUID(),
            exportLayer: (targetLayerId) => engine.exportBakedLayerBlob(targetLayerId, { includeDisabled: true }),
            getImage: getGalleryImageByName,
            isGuardCurrent: (guard) => engine.isLayerExportGuardCurrent(guard),
            makeDurable: makeImageDurable,
            runGraph: (options) => runUtilityGraph({ ...options, hub: socketHub }),
            saveToGallery: saveImageToGallery,
            touchGallery: (targetProjectId) =>
              dispatch({ projectId: targetProjectId, type: 'touchGalleryImagesRefresh' }),
            uploadIntermediate: async (blob, uploadSignal) => {
              if (uploadSignal?.aborted) {
                throw new DOMException('Layer workflow aborted', 'AbortError');
              }
              const uploaded = await executorDeps.uploadImage(blob);
              if (uploadSignal?.aborted) {
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
          signal,
          templatesSnapshot: availability.templatesSnapshot,
        });
        if (result.status === 'completed') {
          notify.success(t(SUCCESS_KEYS[destination]));
          engine.cancelWorkflowOperation();
        } else if (result.status === 'failed') {
          setError(t(getLayerWorkflowFailureKey(result.stage), { message: result.message }));
        } else {
          setError(t(FAILURE_KEYS[result.status]));
        }
      });
      if (operationResult === 'error') {
        const operation = engine.canvasOperations.getSnapshot();
        setError(operation.status === 'active' ? operation.error : t('widgets.layers.runWorkflow.failed'));
      }
    } catch (cause) {
      setError(t('widgets.layers.runWorkflow.failed', { message: messageOf(cause) }));
    } finally {
      setIsRunning(false);
    }
  }, [availability, dispatch, engine, isRunning, layerId, notify, projectId, selection, t]);
  const handleRun = useCallback(() => void run(), [run]);

  return (
    <HStack align="center" gap="2" maxW="calc(100vw - 2rem)" overflowX="auto">
      <NativeSelect.Root disabled={!eligibility.canEdit} size="xs" w="12rem">
        <NativeSelect.Field
          aria-label={t('widgets.layers.runWorkflow.output')}
          value={selection.output ? layerWorkflowBindingKey(selection.output) : ''}
          onChange={handleOutput}
        >
          {availability.outputs.map((binding) => (
            <option key={layerWorkflowBindingKey(binding)} value={layerWorkflowBindingKey(binding)}>
              {binding.label}
            </option>
          ))}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
      <NativeSelect.Root disabled={!eligibility.canEdit} size="xs" w="12rem">
        <NativeSelect.Field
          aria-label={t('widgets.layers.runWorkflow.input')}
          value={selection.input ? layerWorkflowBindingKey(selection.input) : ''}
          onChange={handleInput}
        >
          {runnableInputs.map((binding) => (
            <option key={layerWorkflowBindingKey(binding)} value={layerWorkflowBindingKey(binding)}>
              {binding.label}
            </option>
          ))}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
      <NativeSelect.Root disabled={!eligibility.canEdit} size="xs" w="9rem">
        <NativeSelect.Field
          aria-label={t('widgets.layers.runWorkflow.destination')}
          value={selection.destination}
          onChange={handleDestination}
        >
          <option value="gallery">{t('widgets.layers.runWorkflow.gallery')}</option>
          <option value="staging">{t('widgets.layers.runWorkflow.staging')}</option>
          <option value="replace">{t('widgets.layers.runWorkflow.replace')}</option>
          <option value="copy-raster">{t('widgets.layers.runWorkflow.copyRaster')}</option>
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
      <Button disabled={!eligibility.canRun} loading={isRunning} size="xs" onClick={handleRun}>
        {t('widgets.layers.runWorkflow.run')}
      </Button>
      <Button disabled={!eligibility.canCancel} size="xs" variant="ghost" onClick={cancel}>
        {t('widgets.layers.runWorkflow.cancel')}
      </Button>
      {error ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {error}
        </Text>
      ) : null}
    </HStack>
  );
};
