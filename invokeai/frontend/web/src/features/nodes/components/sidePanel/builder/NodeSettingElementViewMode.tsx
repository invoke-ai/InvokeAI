import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import {
  useIsNodeSettingApplicable,
  useNodeSetting,
  useNodeSettingDefaultLabel,
} from 'features/nodes/hooks/useNodeSetting';
import type { NodeSettingElement } from 'features/nodes/types/workflow';
import { NODE_SETTING_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  pb: 2,
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    minW: 32,
  },
};

export const NodeSettingElementViewMode = memo(({ el }: { el: NodeSettingElement }) => {
  const { id, data } = el;
  const containerCtx = useContainerContext();
  const defaultLabel = useNodeSettingDefaultLabel(data.setting);
  const isApplicable = useIsNodeSettingApplicable();
  const isAdmin = useIsAdmin();
  const { isChecked, onChange, isConnected } = useNodeSetting(data.nodeId, data.setting);

  // Settings that no longer apply to their node would render as no-op toggles, so hide them. Node-cache control is
  // admin-only (single-user mode counts as admin), matching the node footer.
  if (!isApplicable || (data.setting === 'use_cache' && !isAdmin)) {
    return null;
  }

  return (
    <Flex id={id} className={NODE_SETTING_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <FormControl flex="1 1 0" orientation="vertical" isDisabled={isConnected}>
        <FormLabel>{data.label || defaultLabel}</FormLabel>
        <Flex w="full" gap={4}>
          {/* An edge drives the value, so the local one is stale - show it, but don't let it be changed here */}
          <Switch isChecked={isChecked} onChange={onChange} isDisabled={isConnected} />
        </Flex>
      </FormControl>
    </Flex>
  );
});
NodeSettingElementViewMode.displayName = 'NodeSettingElementViewMode';
