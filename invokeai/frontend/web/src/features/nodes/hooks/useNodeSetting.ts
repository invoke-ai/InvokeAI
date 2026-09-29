import { useAppDispatch } from 'app/store/storeHooks';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useNodeIsIntermediate } from 'features/nodes/hooks/useNodeIsIntermediate';
import { useNodeTemplateSafe } from 'features/nodes/hooks/useNodeTemplateSafe';
import { useUseCache } from 'features/nodes/hooks/useUseCache';
import { nodeIsIntermediateChanged, nodeUseCacheChanged } from 'features/nodes/store/nodesSlice';
import { getHasNodeFooter } from 'features/nodes/types/invocation';
import type { NodeAttributeFieldName } from 'features/nodes/types/nodeAttributeFields';
import type { NodeSettingName } from 'features/nodes/types/workflow';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The backend field each node setting maps to. Note that `save_to_gallery` is the inverse of `is_intermediate`.
 */
export const NODE_SETTING_FIELD_NAMES: Record<NodeSettingName, NodeAttributeFieldName> = {
  use_cache: 'use_cache',
  save_to_gallery: 'is_intermediate',
};

/**
 * The label shown for a node setting when the workflow author has not provided their own.
 */
export const useNodeSettingDefaultLabel = (setting: NodeSettingName): string => {
  const { t } = useTranslation();
  return useMemo(
    () => (setting === 'use_cache' ? t('invocationCache.useCache') : t('nodes.saveToGallery')),
    [setting, t]
  );
};

/**
 * Whether a user may see and change a node setting.
 *
 * `use_cache` drives the process-global invocation cache, so it is admin-only (single-user mode counts as admin).
 *
 * Exported separately from the hook so the predicate can be unit-tested directly.
 */
export const getIsNodeSettingPermitted = (setting: NodeSettingName, isAdmin: boolean): boolean =>
  setting !== 'use_cache' || isAdmin;

/**
 * Whether the current user may see and change a node setting.
 *
 * This is the single gate for every path that surfaces a setting - the node footer, and the form builder's view and
 * edit modes. The add-to-form and drag-into-form paths live inside the footer control, so they are gated by it too.
 */
export const useIsNodeSettingPermitted = (setting: NodeSettingName): boolean => {
  const isAdmin = useIsAdmin();
  return getIsNodeSettingPermitted(setting, isAdmin);
};

/**
 * Whether a node setting applies to the node it belongs to. A setting applies exactly when the node renders a
 * footer, since that is where the setting lives on the node itself.
 *
 * Must be used within an `InvocationNodeContextProvider`.
 */
export const useIsNodeSettingApplicable = (): boolean => {
  const template = useNodeTemplateSafe();
  return useMemo(() => (template ? getHasNodeFooter(template) : false), [template]);
};

/**
 * Provides the checked state and change handler for a node setting, plus whether an edge is driving it.
 *
 * When connected, the node runs with the edge's value, so the local value is stale and must not be presented as
 * something the user can change.
 *
 * Must be used within an `InvocationNodeContextProvider`.
 */
export const useNodeSetting = (nodeId: string, setting: NodeSettingName) => {
  const dispatch = useAppDispatch();
  const useCache = useUseCache();
  const isIntermediate = useNodeIsIntermediate();
  const isConnected = useInputFieldIsConnected(NODE_SETTING_FIELD_NAMES[setting]);

  const isChecked = useMemo(
    () => (setting === 'use_cache' ? useCache : !isIntermediate),
    [isIntermediate, setting, useCache]
  );

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      if (setting === 'use_cache') {
        dispatch(nodeUseCacheChanged({ nodeId, useCache: e.target.checked }));
      } else {
        dispatch(nodeIsIntermediateChanged({ nodeId, isIntermediate: !e.target.checked }));
      }
    },
    [dispatch, nodeId, setting]
  );

  return { isChecked, onChange, isConnected };
};
