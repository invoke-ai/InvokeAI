import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { DividerElement } from 'features/nodes/types/workflow';
import { DIVIDER_CLASS_NAME, isDividerElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  bg: 'base.700',
  flexShrink: 0,
  '&[data-layout="column"]': {
    width: '100%',
    height: '1px',
  },
  '&[data-layout="row"]': {
    height: '100%',
    width: '1px',
    minH: 32,
  },
};

export const DividerElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isDividerElement(el)) {
    return;
  }

  if (mode === 'view') {
    return <DividerElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <DividerElementComponentEditMode el={el} />;
});

DividerElementComponent.displayName = 'DividerElementComponent';

const DividerElementComponentViewMode = memo(({ el }: { el: DividerElement }) => {
  const container = useContainerContext();
  const { id } = el;

  return (
    <Flex
      id={id}
      className={DIVIDER_CLASS_NAME}
      sx={sx}
      data-layout={
        // When there is no container, the layout is column by default
        container?.layout || 'column'
      }
    />
  );
});

DividerElementComponentViewMode.displayName = 'DividerElementComponentViewMode';

const DividerElementComponentEditMode = memo(({ el }: { el: DividerElement }) => {
  const container = useContainerContext();
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex
        id={id}
        className={DIVIDER_CLASS_NAME}
        sx={sx}
        data-layout={
          // When there is no container, the layout is column by default
          container?.layout || 'column'
        }
      />
    </FormElementEditModeWrapper>
  );
});

DividerElementComponentEditMode.displayName = 'DividerElementComponentEditMode';
