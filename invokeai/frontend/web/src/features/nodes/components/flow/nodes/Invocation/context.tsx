import { useStore } from '@nanostores/react';
import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodeFieldElements, selectNodes } from 'features/nodes/store/selectors';
import type { InvocationNode, InvocationTemplate } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';

type InvocationNodeContextValue = {
  nodeId: string;

  selectNodeSafe: Selector<RootState, InvocationNode | null>;
  selectNodeDataSafe: Selector<RootState, InvocationNode['data'] | null>;
  selectNodeTypeSafe: Selector<RootState, string | null>;
  selectNodeTemplateSafe: Selector<RootState, InvocationTemplate | null>;
  selectNodeInputsSafe: Selector<RootState, InvocationNode['data']['inputs'] | null>;

  buildSelectInputFieldSafe: (
    fieldName: string
  ) => Selector<RootState, InvocationNode['data']['inputs'][string] | null>;
  buildSelectInputFieldTemplateSafe: (
    fieldName: string
  ) => Selector<RootState, InvocationTemplate['inputs'][string] | null>;
  buildSelectOutputFieldTemplateSafe: (
    fieldName: string
  ) => Selector<RootState, InvocationTemplate['outputs'][string] | null>;
  buildSelectIsInputFieldAddedToForm: (fieldName: string) => Selector<RootState, boolean>;

  selectNodeOrThrow: Selector<RootState, InvocationNode>;
  selectNodeDataOrThrow: Selector<RootState, InvocationNode['data']>;
  selectNodeTypeOrThrow: Selector<RootState, string>;
  selectNodeTemplateOrThrow: Selector<RootState, InvocationTemplate>;
  selectNodeInputsOrThrow: Selector<RootState, InvocationNode['data']['inputs']>;

  buildSelectInputFieldOrThrow: (fieldName: string) => Selector<RootState, InvocationNode['data']['inputs'][string]>;
  buildSelectInputFieldTemplateOrThrow: (
    fieldName: string
  ) => Selector<RootState, InvocationTemplate['inputs'][string]>;
  buildSelectOutputFieldTemplateOrThrow: (
    fieldName: string
  ) => Selector<RootState, InvocationTemplate['outputs'][string]>;

  buildSelectIsInputFieldConnected: (fieldName: string) => Selector<RootState, boolean>;
  selectNodeNeedsUpdate: Selector<RootState, boolean>;
};

const InvocationNodeContext = createContext<InvocationNodeContextValue | null>(null);

const getSelectorFromCache = <T extends Selector>(cache: Map<string, Selector>, key: string, fallback: () => T): T => {
  let selector = cache.get(key);
  if (!selector) {
    selector = fallback();
    cache.set(key, selector);
  }
  return selector as T;
};

export const InvocationNodeContextProvider = memo(({ nodeId, children }: PropsWithChildren<{ nodeId: string }>) => {
  const templates = useStore($templates);

  const value = useMemo(() => {
    /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
    const cache: Map<string, Selector<RootState, any>> = new Map();

    const selectNodeSafe = getSelectorFromCache(cache, 'selectNodeSafe', () =>
      createSelector(selectNodes, (nodes) => {
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
      createSelector(selectNodes, (nodes) => {
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
        createSelector(selectEdges, (edges) => {
          return edges.some((edge) => {
            return edge.target === nodeId && edge.targetHandle === fieldName;
          });
        })
      );

    const buildSelectIsInputFieldAddedToForm = (fieldName: string) =>
      getSelectorFromCache(cache, `buildSelectIsInputFieldAddedToForm-${fieldName}`, () =>
        createSelector(selectNodeFieldElements, (nodeFieldElements) => {
          return nodeFieldElements.some(
            (el) => el.data.fieldIdentifier.nodeId === nodeId && el.data.fieldIdentifier.fieldName === fieldName
          );
        })
      );

    const selectNodeNeedsUpdate = getSelectorFromCache(cache, 'selectNodeNeedsUpdate', () =>
      createSelector([selectNodeDataSafe, selectNodeTemplateSafe], (data, template) => {
        if (!data || !template) {
          return false; // If there's no data or template, no update is possible
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
    } satisfies InvocationNodeContextValue;
  }, [nodeId, templates]);

  return <InvocationNodeContext.Provider value={value}>{children}</InvocationNodeContext.Provider>;
});
InvocationNodeContextProvider.displayName = 'InvocationNodeContextProvider';

export const useInvocationNodeContext = () => {
  const context = useContext(InvocationNodeContext);
  if (!context) {
    throw new Error('useInvocationNodeContext must be used within an InvocationNodeProvider');
  }
  return context;
};
