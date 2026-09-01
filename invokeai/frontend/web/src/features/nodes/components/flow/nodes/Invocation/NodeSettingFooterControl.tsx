import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Checkbox, Flex, FormControl, FormLabel, Spacer } from '@invoke-ai/ui-library';
import { InputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldHandle';
import { useNodeSettingDnd } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { useInputFieldTemplateSafe } from 'features/nodes/hooks/useInputFieldTemplateSafe';
import {
  NODE_SETTING_FIELD_NAMES,
  useIsNodeSettingPermitted,
  useNodeSetting,
  useNodeSettingDefaultLabel,
} from 'features/nodes/hooks/useNodeSetting';
import { NO_DRAG_CLASS, NO_FIT_ON_DOUBLE_CLICK_CLASS, NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { NodeSettingName } from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';

import { NodeSettingAddRemoveFormRoot } from './NodeSettingAddRemoveFormRoot';

// Mirrors `InputFieldWrapper`: one row per field. The connection handle is absolutely positioned against this row and
// keeps its static vertical position, so it lines up with the label it belongs to.
const sx: SystemStyleObject = {
  position: 'relative',
  w: 'full',
  minH: 6,
  alignItems: 'center',
  // Clear the half of the handle that overlaps the node's interior
  ps: 2,
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
};

const formControlSx: SystemStyleObject = { w: 'full', alignItems: 'center', gap: 2 };

type Props = {
  nodeId: string;
  setting: NodeSettingName;
};

/**
 * A node attribute toggle in the node footer.
 *
 * The value lives on the node (`data.useCache` / `data.isIntermediate`), but the underlying field is declared on the
 * backend as a connectable input, so this also hosts the field's connection handle. When an edge drives the value the
 * checkbox is dropped and only the label and handle remain, matching how a connected input field renders.
 */
export const NodeSettingFooterControl = memo(({ nodeId, setting }: Props) => {
  const fieldName = NODE_SETTING_FIELD_NAMES[setting];
  const label = useNodeSettingDefaultLabel(setting);
  const { isChecked, onChange, isConnected } = useNodeSetting(nodeId, setting);
  // The template is what the handle needs. It should always be present, but a node whose template failed to parse
  // this field must not take the whole footer down with it.
  const fieldTemplate = useInputFieldTemplateSafe(fieldName);
  const isPermitted = useIsNodeSettingPermitted(setting);
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const isDragging = useNodeSettingDnd(nodeId, setting, draggableRef, dragHandleRef);

  // Hiding the control here also removes the add-to-form button and the drag handle it hosts, so an unpermitted
  // setting cannot be put into the form in the first place.
  if (!isPermitted) {
    return null;
  }

  return (
    <Flex ref={draggableRef} sx={sx} data-is-dragging={isDragging}>
      <FormControl className={`${NO_FIT_ON_DOUBLE_CLICK_CLASS} ${NO_PAN_CLASS}`} sx={formControlSx}>
        <Flex className={NO_DRAG_CLASS} ref={dragHandleRef}>
          <FormLabel m={0}>{label}</FormLabel>
        </Flex>
        <Spacer />
        <NodeSettingAddRemoveFormRoot nodeId={nodeId} setting={setting} />
        {!isConnected && <Checkbox className={NO_PAN_CLASS} onChange={onChange} isChecked={isChecked} />}
      </FormControl>
      {fieldTemplate && <InputFieldHandle nodeId={nodeId} fieldName={fieldName} />}
    </Flex>
  );
});

NodeSettingFooterControl.displayName = 'NodeSettingFooterControl';
