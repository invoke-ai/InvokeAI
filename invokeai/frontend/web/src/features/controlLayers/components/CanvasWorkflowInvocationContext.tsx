import { useStore } from '@nanostores/react';
import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodeFieldElements, selectNodes } from 'features/nodes/store/selectors';
import { InvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { InvocationNode, InvocationTemplate } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

/**
 * Provides InvocationNodeContext for canvas workflow nodes.
 *
 * This is a wrapper around InvocationNodeContextProvider that redirects
 * node selectors to use the canvasWorkflowNodes slice instead of the nodes slice.
 * This allows all existing field components to work without modification.
 */

const getSelectorFromCache = <T extends Selector>(cache: Map<string, Selector>, key: string, fallback: () => T): T => {
  let selector = cache.get(key);
  if (!selector) {
    selector = fallback();
    cache.set(key, selector);
  }
  return selector as T;
};

// Create custom selectors that read from canvasWorkflowNodes instead of nodes
const selectCanvasWorkflowNodes = (state: RootState) => state.canvasWorkflowNodes.nodes;
const selectCanvasWorkflowEdges = (state: RootState) => state.canvasWorkflowNodes.edges;
const selectCanvasWorkflowNodeFieldElements = (state: RootState) => {
  const form = state.canvasWorkflowNodes.form;
  return Object.values(form.elements).filter((el) => el.type === 'node-field');
};

export const CanvasWorkflowInvocationNodeContextProvider = memo(
  ({ nodeId, children }: PropsWithChildren<{ nodeId: string }>) => {
    const templates = useStore($templates);

    const value = useMemo(() => {
      const cache: Map<string, Selector<RootState, any>> = new Map();

      const selectNodeSafe = getSelectorFromCache(cache, 'selectNodeSafe', () =>
        createSelector(selectCanvasWorkflowNodes, (nodes) => {
          return (nodes.find(({ id, type }) => type === 'invocation' && id === nodeId) ?? null) as InvocationNode | null;
        })
      );
      const selectNodeDataSafe = getSelectorFromCache(cache, 'selectNodeDataSafe', () =>
        createSelector(selectNodeSafe, (node) => {
          return node?.data ?? null;
        })
      );
      const selectNodeTypeSafe = getSelectorFromCache(cache, 'selectNodeTypeSafe', () =>
        createSelector(selectNodeDataSafe, (data) => {
          return data?.type ?? null;
        })
      );
      const selectNodeTemplateSafe = getSelectorFromCache(cache, 'selectNodeTemplateSafe', () =>
        createSelector(selectNodeTypeSafe, (type) => {
          return type ? (templates[type] ?? null) : null;
        })
      );
      const selectNodeInputsSafe = getSelectorFromCache(cache, 'selectNodeInputsSafe', () =>
        createSelector(selectNodeDataSafe, (data) => {
          return data?.inputs ?? null;
        })
      );
      const buildSelectInputFieldSafe = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectInputFieldSafe-${fieldName}`, () =>
          createSelector(selectNodeInputsSafe, (inputs) => {
            return inputs?.[fieldName] ?? null;
          })
        );
      const buildSelectInputFieldTemplateSafe = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectInputFieldTemplateSafe-${fieldName}`, () =>
          createSelector(selectNodeTemplateSafe, (template) => {
            return template?.inputs?.[fieldName] ?? null;
          })
        );
      const buildSelectOutputFieldTemplateSafe = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectOutputFieldTemplateSafe-${fieldName}`, () =>
          createSelector(selectNodeTemplateSafe, (template) => {
            return template?.outputs?.[fieldName] ?? null;
          })
        );

      const selectNodeOrThrow = getSelectorFromCache(cache, 'selectNodeOrThrow', () =>
        createSelector(selectCanvasWorkflowNodes, (nodes) => {
          const node = nodes.find(({ id, type }) => type === 'invocation' && id === nodeId) as InvocationNode | undefined;
          if (node === undefined) {
            throw new Error(`Cannot find node with id ${nodeId}`);
          }
          return node;
        })
      );
      const selectNodeDataOrThrow = getSelectorFromCache(cache, 'selectNodeDataOrThrow', () =>
        createSelector(selectNodeOrThrow, (node) => {
          return node.data;
        })
      );
      const selectNodeTypeOrThrow = getSelectorFromCache(cache, 'selectNodeTypeOrThrow', () =>
        createSelector(selectNodeDataOrThrow, (data) => {
          return data.type;
        })
      );
      const selectNodeTemplateOrThrow = getSelectorFromCache(cache, 'selectNodeTemplateOrThrow', () =>
        createSelector(selectNodeTypeOrThrow, (type) => {
          const template = templates[type];
          if (template === undefined) {
            throw new Error(`Cannot find template for node with id ${nodeId} with type ${type}`);
          }
          return template;
        })
      );
      const selectNodeInputsOrThrow = getSelectorFromCache(cache, 'selectNodeInputsOrThrow', () =>
        createSelector(selectNodeDataOrThrow, (data) => {
          return data.inputs;
        })
      );
      const buildSelectInputFieldOrThrow = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectInputFieldOrThrow-${fieldName}`, () =>
          createSelector(selectNodeInputsOrThrow, (inputs) => {
            const field = inputs[fieldName];
            if (field === undefined) {
              console.error(`[CanvasWorkflowContext] Cannot find input field with name ${fieldName} in node ${nodeId}. Available fields:`, Object.keys(inputs));
              throw new Error(`Cannot find input field with name ${fieldName} in node ${nodeId}`);
            }
            return field;
          })
        );
      const buildSelectInputFieldTemplateOrThrow = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectInputFieldTemplateOrThrow-${fieldName}`, () =>
          createSelector(selectNodeTemplateOrThrow, (template) => {
            const fieldTemplate = template.inputs[fieldName];
            if (fieldTemplate === undefined) {
              throw new Error(`Cannot find input field template with name ${fieldName} in node ${nodeId}`);
            }
            return fieldTemplate;
          })
        );
      const buildSelectOutputFieldTemplateOrThrow = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectOutputFieldTemplateOrThrow-${fieldName}`, () =>
          createSelector(selectNodeTemplateOrThrow, (template) => {
            const fieldTemplate = template.outputs[fieldName];
            if (fieldTemplate === undefined) {
              throw new Error(`Cannot find output field template with name ${fieldName} in node ${nodeId}`);
            }
            return fieldTemplate;
          })
        );

      const buildSelectIsInputFieldConnected = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectIsInputFieldConnected-${fieldName}`, () =>
          createSelector(selectCanvasWorkflowEdges, (edges) => {
            return edges.some((edge) => {
              return edge.target === nodeId && edge.targetHandle === fieldName;
            });
          })
        );

      const buildSelectIsInputFieldAddedToForm = (fieldName: string) =>
        getSelectorFromCache(cache, `buildSelectIsInputFieldAddedToForm-${fieldName}`, () =>
          createSelector(selectCanvasWorkflowNodeFieldElements, (nodeFieldElements) => {
            return nodeFieldElements.some(
              (el: any) => el.data.fieldIdentifier.nodeId === nodeId && el.data.fieldIdentifier.fieldName === fieldName
            );
          })
        );

      const selectNodeNeedsUpdate = getSelectorFromCache(cache, 'selectNodeNeedsUpdate', () =>
        createSelector([selectNodeDataSafe, selectNodeTemplateSafe], (data, template) => {
          if (!data || !template) {
            return false;
          }
          return getNeedsUpdate(data, template);
        })
      );

      return {
        nodeId,

        selectNodeSafe,
        selectNodeDataSafe,
        selectNodeTypeSafe,
        selectNodeTemplateSafe,
        selectNodeInputsSafe,

        buildSelectInputFieldSafe,
        buildSelectInputFieldTemplateSafe,
        buildSelectOutputFieldTemplateSafe,
        buildSelectIsInputFieldAddedToForm,

        selectNodeOrThrow,
        selectNodeDataOrThrow,
        selectNodeTypeOrThrow,
        selectNodeTemplateOrThrow,
        selectNodeInputsOrThrow,

        buildSelectInputFieldOrThrow,
        buildSelectInputFieldTemplateOrThrow,
        buildSelectOutputFieldTemplateOrThrow,

        buildSelectIsInputFieldConnected,
        selectNodeNeedsUpdate,
      };
    }, [nodeId, templates]);

    return <InvocationNodeContext.Provider value={value}>{children}</InvocationNodeContext.Provider>;
  }
);
CanvasWorkflowInvocationNodeContextProvider.displayName = 'CanvasWorkflowInvocationNodeContextProvider';