import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, FormControl, Switch, Text } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { useFormElementDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { FormElementNodeOverlay } from 'features/nodes/components/sidePanel/builder/FormElementNodeOverlay';
import { NodeSettingElementLabelEditable } from 'features/nodes/components/sidePanel/builder/NodeSettingElementLabelEditable';
import { useIsNodeSettingApplicable, useNodeSetting } from 'features/nodes/hooks/useNodeSetting';
import type { NodeSettingElement } from 'features/nodes/types/workflow';
import { NODE_SETTING_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const sx: SystemStyleObject = {
  position: 'relative',
  borderRadius: 'base',
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
  },
  flexDir: 'column',
};

export const NodeSettingElementEditMode = memo(({ el }: { el: NodeSettingElement }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(el.id, draggableRef, dragHandleRef);
  const containerCtx = useContainerContext();
  const { id } = el;

  return (
    <Flex
      ref={draggableRef}
      id={id}
      className={NODE_SETTING_CLASS_NAME}
      sx={sx}
      data-parent-layout={containerCtx.layout}
    >
      <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
      <FormElementEditModeContent data-is-dragging={isDragging} p={4}>
        <NodeSettingElementEditModeContent el={el} />
      </FormElementEditModeContent>
      <FormElementNodeOverlay nodeId={el.data.nodeId} />
      <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
    </Flex>
  );
});
NodeSettingElementEditMode.displayName = 'NodeSettingElementEditMode';

const NodeSettingElementEditModeContent = memo(({ el }: { el: NodeSettingElement }) => {
  const { t } = useTranslation();
  const { data } = el;
  const isApplicable = useIsNodeSettingApplicable();
  const { isChecked, onChange, isConnected } = useNodeSetting(data.nodeId, data.setting);

  if (!isApplicable) {
    // The element is hidden in view mode, so surface why it won't render while editing.
    return (
      <Text fontWeight="semibold" color="error.300">
        {t('workflows.builder.nodeSettingNotApplicable')}
      </Text>
    );
  }

  return (
    <FormControl flex="1 1 0" orientation="vertical" isDisabled={isConnected}>
      <NodeSettingElementLabelEditable el={el} />
      <Flex w="full" gap={4}>
        {/* An edge drives the value, so the local one is stale - show it, but don't let it be changed here */}
        <Switch isChecked={isChecked} onChange={onChange} isDisabled={isConnected} />
      </Flex>
    </FormControl>
  );
});
NodeSettingElementEditModeContent.displayName = 'NodeSettingElementEditModeContent';
