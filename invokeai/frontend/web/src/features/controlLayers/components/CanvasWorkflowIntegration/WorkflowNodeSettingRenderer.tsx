import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { getNodeSettingKey } from 'features/controlLayers/components/CanvasWorkflowIntegration/nodeSettingOverrides';
import {
  canvasWorkflowIntegrationNodeSettingChanged,
  selectCanvasWorkflowIntegrationNodeSettingValues,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import {
  getIsNodeSettingPermitted,
  NODE_SETTING_FIELD_NAMES,
  useNodeSettingDefaultLabel,
} from 'features/nodes/hooks/useNodeSetting';
import { $templates } from 'features/nodes/store/nodesSlice';
import { getHasNodeFooter } from 'features/nodes/types/invocation';
import type { NodeSettingElement } from 'features/nodes/types/workflow';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

const log = logger('canvas-workflow-integration');

/**
 * A Use Cache / Save To Gallery control in the canvas form preview.
 *
 * Unlike a node field, a node setting has no entry in `node.data.inputs` - its value is a node attribute. The preview
 * therefore keeps its own override in `canvasWorkflowIntegrationSlice.nodeSettingValues`, which the executor applies
 * to the invocation it builds (see `resolveNodeSettings`). Without that the control would be inert, since the graph
 * is built from the workflow as saved.
 */
export const WorkflowNodeSettingRenderer = memo(({ el }: { el: NodeSettingElement }) => {
  const dispatch = useAppDispatch();
  const { nodeId, setting } = el.data;
  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const nodeSettingValues = useAppSelector(selectCanvasWorkflowIntegrationNodeSettingValues);
  const templates = useStore($templates);
  const isAdmin = useIsAdmin();
  const defaultLabel = useNodeSettingDefaultLabel(setting);
  const settingKey = getNodeSettingKey(nodeId, setting);

  const { data: workflow } = useGetWorkflowQuery(selectedWorkflowId!, { skip: !selectedWorkflowId });

  const node = useMemo(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return workflow?.workflow.nodes.find((n: any) => n.data?.id === nodeId);
  }, [workflow, nodeId]);

  // An edge drives the value, so whatever is set here would be overwritten at run time.
  const isConnected = useMemo(() => {
    const fieldName = NODE_SETTING_FIELD_NAMES[setting];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return Boolean(workflow?.workflow.edges.some((e: any) => e.target === nodeId && e.targetHandle === fieldName));
  }, [workflow, nodeId, setting]);

  const isChecked = useMemo(() => {
    const override = nodeSettingValues?.[settingKey];
    if (override !== undefined) {
      return override;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const data = node?.data as any;
    return setting === 'use_cache' ? (data?.useCache ?? true) : !(data?.isIntermediate ?? false);
  }, [nodeSettingValues, settingKey, node, setting]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(canvasWorkflowIntegrationNodeSettingChanged({ settingKey, value: e.target.checked }));
    },
    [dispatch, settingKey]
  );

  if (!node) {
    log.warn({ nodeId, setting }, 'Node for node setting not found');
    return null;
  }

  // Same gate as the workflow editor: the cache is a process-global, admin-only policy.
  if (!getIsNodeSettingPermitted(setting, isAdmin)) {
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const template = templates[(node.data as any).type];
  if (!template || !getHasNodeFooter(template)) {
    // The setting does not apply to this node - it would render as a no-op toggle, exactly as in the workflow editor.
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (setting === 'save_to_gallery' && (node.data as any).type === 'canvas_output') {
    // Canvas output images go to the staging area, never straight to the gallery, so this one is not the user's to
    // set here - `resolveNodeSettings` forces it.
    return null;
  }

  return (
    <FormControl isDisabled={isConnected}>
      <FormLabel>{el.data.label || defaultLabel}</FormLabel>
      {/* An edge drives the value, so the local one is stale - show it, but don't let it be changed here */}
      <Switch isChecked={isChecked} onChange={onChange} isDisabled={isConnected} />
    </FormControl>
  );
});

WorkflowNodeSettingRenderer.displayName = 'WorkflowNodeSettingRenderer';
