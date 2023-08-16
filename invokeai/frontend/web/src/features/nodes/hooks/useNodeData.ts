import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map, some } from 'lodash-es';
import { useMemo } from 'react';
import { FOOTER_FIELDS, IMAGE_FIELDS } from '../types/constants';
import { isInvocationNode } from '../types/types';

const KIND_MAP = {
  input: 'inputs' as const,
  output: 'outputs' as const,
};

export const useNodeTemplate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
          return nodeTemplate;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};

export const useNodeData = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          return node?.data;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const nodeData = useAppSelector(selector);

  return nodeData;
};

export const useFieldData = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          return node?.data.inputs[fieldName];
        },
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const fieldData = useAppSelector(selector);

  return fieldData;
};

export const useFieldType = (
  nodeId: string,
  fieldName: string,
  kind: 'input' | 'output'
) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          return node?.data[KIND_MAP[kind]][fieldName]?.type;
        },
        defaultSelectorOptions
      ),
    [fieldName, kind, nodeId]
  );

  const fieldType = useAppSelector(selector);

  return fieldType;
};

export const useFieldNames = (nodeId: string, kind: 'input' | 'output') => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return [];
          }
          return map(node.data[KIND_MAP[kind]], (field) => field.name).filter(
            (fieldName) => fieldName !== 'is_intermediate'
          );
        },
        defaultSelectorOptions
      ),
    [kind, nodeId]
  );

  const fieldNames = useAppSelector(selector);
  return fieldNames;
};

export const useWithFooter = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          return some(node.data.outputs, (output) =>
            FOOTER_FIELDS.includes(output.type)
          );
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const withFooter = useAppSelector(selector);
  return withFooter;
};

export const useHasImageOutput = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          return some(node.data.outputs, (output) =>
            IMAGE_FIELDS.includes(output.type)
          );
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const hasImageOutput = useAppSelector(selector);
  return hasImageOutput;
};

export const useIsIntermediate = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          return Boolean(node.data.inputs.is_intermediate?.value);
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const is_intermediate = useAppSelector(selector);
  return is_intermediate;
};

export const useNodeLabel = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }

          return node.data.label;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const label = useAppSelector(selector);
  return label;
};

export const useNodeTemplateTitle = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return false;
          }
          const nodeTemplate = node
            ? nodes.nodeTemplates[node.data.type]
            : undefined;

          return nodeTemplate?.title;
        },
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const title = useAppSelector(selector);
  return title;
};

export const useFieldTemplate = (
  nodeId: string,
  fieldName: string,
  kind: 'input' | 'output'
) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];
          return nodeTemplate?.[KIND_MAP[kind]][fieldName];
        },
        defaultSelectorOptions
      ),
    [fieldName, kind, nodeId]
  );

  const fieldTemplate = useAppSelector(selector);

  return fieldTemplate;
};

export const useDoesInputHaveValue = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const node = nodes.nodes.find((node) => node.id === nodeId);
          if (!isInvocationNode(node)) {
            return;
          }
          return Boolean(node?.data.inputs[fieldName]?.value);
        },
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const doesFieldHaveValue = useAppSelector(selector);

  return doesFieldHaveValue;
};
