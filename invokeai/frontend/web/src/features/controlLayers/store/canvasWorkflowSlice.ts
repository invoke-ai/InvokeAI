import type { PayloadAction } from '@reduxjs/toolkit';
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import { $templates, getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import type { Templates } from 'features/nodes/store/types';
import type { StatefulFieldValue } from 'features/nodes/types/field';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import { serializeError } from 'serialize-error';
import { workflowsApi } from 'services/api/endpoints/workflows';
import { z } from 'zod';

const log = logger('canvas');

const zCanvasWorkflowState = z.object({
  selectedWorkflowId: z.string().nullable(),
  workflow: zWorkflowV3.nullable(),
  inputNodeId: z.string().nullable(),
  outputNodeId: z.string().nullable(),
  fieldValues: z.record(z.string(), z.any()),
  status: z.enum(['idle', 'loading', 'succeeded', 'failed']),
  error: z.string().nullable(),
});

export type CanvasWorkflowState = z.infer<typeof zCanvasWorkflowState>;

const getInitialState = (): CanvasWorkflowState => ({
  selectedWorkflowId: null,
  workflow: null,
  inputNodeId: null,
  outputNodeId: null,
  fieldValues: {},
  status: 'idle',
  error: null,
});

type ValidateResult = {
  inputNodeId: string;
  outputNodeId: string;
};

const INPUT_TAG = 'canvas-workflow-input';
const OUTPUT_TAG = 'canvas-workflow-output';

const validateCanvasWorkflow = (workflow: WorkflowV3, templates: Templates): ValidateResult => {
  const invocationNodes = workflow.nodes.filter(
    (node): node is WorkflowV3['nodes'][number] => node.type === 'invocation'
  );

  const inputNodes = invocationNodes.filter((node) => {
    const template = templates[node.data.type];
    return Boolean(template && template.tags.includes(INPUT_TAG));
  });

  const outputNodes = invocationNodes.filter((node) => {
    const template = templates[node.data.type];
    return Boolean(template && template.tags.includes(OUTPUT_TAG));
  });

  if (inputNodes.length !== 1) {
    throw new Error('A canvas workflow must include exactly one input node.');
  }

  if (outputNodes.length !== 1) {
    throw new Error('A canvas workflow must include exactly one output node.');
  }

  const inputNode = inputNodes[0]!;
  const outputNode = outputNodes[0]!;

  const inputTemplate = templates[inputNode.data.type];
  if (!inputTemplate) {
    throw new Error(`Input node template "${inputNode.data.type}" not found.`);
  }
  if (!('image' in inputTemplate.inputs)) {
    throw new Error('Canvas input node must expose an image field.');
  }

  const outputTemplate = templates[outputNode.data.type];
  if (!outputTemplate) {
    throw new Error(`Output node template "${outputNode.data.type}" not found.`);
  }
  if (!('image' in outputTemplate.inputs)) {
    throw new Error('Canvas output node must accept an image input field named "image".');
  }

  return { inputNodeId: inputNode.id, outputNodeId: outputNode.id };
};

export const selectCanvasWorkflow = createAsyncThunk<
  {
    workflowId: string;
    workflow: WorkflowV3;
    inputNodeId: string;
    outputNodeId: string;
    fieldValues: Record<string, StatefulFieldValue>;
  },
  string,
  { rejectValue: string }
>('canvasWorkflow/select', async (workflowId, { dispatch, rejectWithValue }) => {
  const request = dispatch(workflowsApi.endpoints.getWorkflow.initiate(workflowId, { subscribe: false }));
  try {
    const result = await request.unwrap();
    const workflow = zWorkflowV3.parse(deepClone(result.workflow));
    const templates = $templates.get();
    if (!Object.keys(templates).length) {
      throw new Error('Invocation templates are not yet available.');
    }
    const { inputNodeId, outputNodeId } = validateCanvasWorkflow(workflow, templates);
    const fieldValues = getFormFieldInitialValues(workflow.form, workflow.nodes);
    return { workflowId: result.workflow_id, workflow, inputNodeId, outputNodeId, fieldValues };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to load workflow.';
    log.error({ error: serializeError(error as Error) }, 'Failed to load canvas workflow');
    return rejectWithValue(message);
  } finally {
    request.unsubscribe();
  }
});

const slice = createSlice({
  name: 'canvasWorkflow',
  initialState: getInitialState(),
  reducers: {
    canvasWorkflowCleared: () => getInitialState(),
    canvasWorkflowFieldValueChanged: (
      state,
      action: PayloadAction<{ elementId: string; value: StatefulFieldValue }>
    ) => {
      state.fieldValues[action.payload.elementId] = action.payload.value;
    },
  },
  extraReducers(builder) {
    builder
      .addCase(selectCanvasWorkflow.pending, (state) => {
        state.status = 'loading';
        state.error = null;
      })
      .addCase(selectCanvasWorkflow.fulfilled, (state, action) => {
        state.selectedWorkflowId = action.payload.workflowId;
        state.workflow = action.payload.workflow;
        state.inputNodeId = action.payload.inputNodeId;
        state.outputNodeId = action.payload.outputNodeId;
        state.fieldValues = action.payload.fieldValues;
        state.status = 'succeeded';
        state.error = null;
      })
      .addCase(selectCanvasWorkflow.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload ?? action.error.message ?? 'Unable to load workflow.';
      });
  },
});

export const { canvasWorkflowCleared, canvasWorkflowFieldValueChanged } = slice.actions;

export const canvasWorkflowSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasWorkflowState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      const parsed = zCanvasWorkflowState.safeParse(state);
      if (!parsed.success) {
        log.warn({ error: parseify(parsed.error) }, 'Failed to migrate canvas workflow state, resetting to defaults');
        return getInitialState();
      }
      return {
        ...parsed.data,
        fieldValues: parsed.data.fieldValues ?? {},
        status: 'idle',
        error: null,
      } satisfies CanvasWorkflowState;
    },
    persistDenylist: ['status', 'error'],
  },
};

export const selectCanvasWorkflowSlice = (state: RootState) => state.canvasWorkflow;

export const selectCanvasWorkflowStatus = (state: RootState) => selectCanvasWorkflowSlice(state).status;

export const selectCanvasWorkflowError = (state: RootState) => selectCanvasWorkflowSlice(state).error;

export const selectCanvasWorkflowSelection = (state: RootState) => selectCanvasWorkflowSlice(state).selectedWorkflowId;

export const selectCanvasWorkflowData = (state: RootState) => selectCanvasWorkflowSlice(state).workflow;

export const selectCanvasWorkflowNodeIds = (state: RootState) => ({
  inputNodeId: selectCanvasWorkflowSlice(state).inputNodeId,
  outputNodeId: selectCanvasWorkflowSlice(state).outputNodeId,
});

export const selectIsCanvasWorkflowActive = (state: RootState) => {
  const sliceState = selectCanvasWorkflowSlice(state);
  return (
    Boolean(sliceState.workflow && sliceState.inputNodeId && sliceState.outputNodeId) &&
    (sliceState.status === 'succeeded' || sliceState.status === 'idle')
  );
};
