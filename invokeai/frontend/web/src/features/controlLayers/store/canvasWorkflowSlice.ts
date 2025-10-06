import type { PayloadAction } from '@reduxjs/toolkit';
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { parseify } from 'common/util/serialize';
import { $templates, getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import type { NodesState, Templates } from 'features/nodes/store/types';
import { getInvocationNodeErrors } from 'features/nodes/store/util/fieldValidators';
import type { StatefulFieldValue } from 'features/nodes/types/field';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { isWorkflowInvocationNode, zWorkflowV3 } from 'features/nodes/types/workflow';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { serializeError } from 'serialize-error';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import { modelsApi } from 'services/api/endpoints/models';
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

const INPUT_TAG = 'canvas-workflow-input';
const OUTPUT_TAG = 'canvas-workflow-output';

const validateCanvasWorkflow = async (
  workflow: unknown,
  templates: Templates,
  checkImageAccess: (name: string) => Promise<boolean>,
  checkBoardAccess: (id: string) => Promise<boolean>,
  checkModelAccess: (key: string) => Promise<boolean>
): Promise<{ workflow: WorkflowV3; inputNodeId: string; outputNodeId: string }> => {
  // First, use the robust validateWorkflow utility to handle parsing, migration, and general validation
  const { workflow: validatedWorkflow, warnings } = await validateWorkflow({
    workflow,
    templates,
    checkImageAccess,
    checkBoardAccess,
    checkModelAccess,
  });

  // Log any warnings from validation
  if (warnings.length > 0) {
    log.warn({ warnings }, 'Canvas workflow validation warnings');
  }

  // Now perform canvas-specific validation
  const invocationNodes = validatedWorkflow.nodes.filter(isWorkflowInvocationNode);

  const inputNodes = invocationNodes.filter((node) => {
    const template = templates[node.data.type];
    return Boolean(template && template.tags.includes(INPUT_TAG));
  });

  const outputNodes = invocationNodes.filter((node) => {
    const template = templates[node.data.type];
    return Boolean(template && template.tags.includes(OUTPUT_TAG));
  });

  if (inputNodes.length === 0) {
    throw new Error('A canvas workflow must include at least one input node with the "canvas-workflow-input" tag.');
  }

  if (inputNodes.length > 1) {
    throw new Error(
      `A canvas workflow must include exactly one input node, but found ${inputNodes.length}. Remove extra input nodes.`
    );
  }

  if (outputNodes.length === 0) {
    throw new Error('A canvas workflow must include at least one output node with the "canvas-workflow-output" tag.');
  }

  if (outputNodes.length > 1) {
    throw new Error(
      `A canvas workflow must include exactly one output node, but found ${outputNodes.length}. Remove extra output nodes.`
    );
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

  // Validate that required fields without connections have values using the existing field validator
  // Create a temporary nodes state for validation - only nodes and edges are used by the validator
  const tempNodesState = {
    nodes: invocationNodes,
    edges: validatedWorkflow.edges,
  } as NodesState;

  for (const node of invocationNodes) {
    const errors = getInvocationNodeErrors(node.id, templates, tempNodesState);

    // Filter out "no image input" errors for the input node - it will be populated by the graph builder
    const relevantErrors = errors.filter((error) => {
      if (error.type === 'field-error' && error.nodeId === inputNode.id && error.fieldName === 'image') {
        return false; // Skip validation for the input node's image field
      }
      return true;
    });

    if (relevantErrors.length > 0) {
      const firstError = relevantErrors[0];
      if (firstError) {
        if (firstError.type === 'field-error') {
          throw new Error(`${firstError.prefix}: ${firstError.issue}`);
        } else {
          throw new Error(`Node "${node.id}": ${firstError.issue}`);
        }
      }
    }
  }

  return { workflow: validatedWorkflow, inputNodeId: inputNode.id, outputNodeId: outputNode.id };
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
    const templates = $templates.get();
    if (!Object.keys(templates).length) {
      throw new Error('Invocation templates are not yet available.');
    }

    // Define access check functions for workflow validation
    const checkImageAccess = async (name: string): Promise<boolean> => {
      const imageRequest = dispatch(imagesApi.endpoints.getImageDTO.initiate(name));
      try {
        await imageRequest.unwrap();
        return true;
      } catch {
        return false;
      } finally {
        imageRequest.unsubscribe();
      }
    };

    const checkBoardAccess = async (id: string): Promise<boolean> => {
      const boardsRequest = dispatch(boardsApi.endpoints.listAllBoards.initiate({}));
      try {
        const boards = await boardsRequest.unwrap();
        return boards.some((board) => board.board_id === id);
      } catch {
        return false;
      } finally {
        boardsRequest.unsubscribe();
      }
    };

    const checkModelAccess = async (key: string): Promise<boolean> => {
      const modelRequest = dispatch(modelsApi.endpoints.getModelConfig.initiate(key));
      try {
        await modelRequest.unwrap();
        return true;
      } catch {
        return false;
      } finally {
        modelRequest.unsubscribe();
      }
    };

    // Use validateWorkflow to parse, migrate, and validate the workflow
    const { workflow, inputNodeId, outputNodeId } = await validateCanvasWorkflow(
      result.workflow,
      templates,
      checkImageAccess,
      checkBoardAccess,
      checkModelAccess
    );
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

export const { canvasWorkflowCleared } = slice.actions;

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
