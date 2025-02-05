import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { DividerElement } from 'features/nodes/types/workflow';
import { DIVIDER_CLASS_NAME, isDividerElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  bg: 'base.700',
  flexShrink: 0,
  '&[data-orientation="horizontal"]': {
    width: '100%',
    height: '1px',
  },
  '&[data-orientation="vertical"]': {
    height: '100%',
    width: '1px',
  },
};

export const DividerElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

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

export const DividerElementComponentViewMode = memo(({ el }: { el: DividerElement }) => {
  const container = useContainerContext();
  const { id } = el;

  return (
    <Flex
      id={id}
      className={DIVIDER_CLASS_NAME}
      sx={sx}
      data-orientation={container?.layout === 'column' ? 'horizontal' : 'vertical'}
    />
  );
});

DividerElementComponentViewMode.displayName = 'DividerElementComponentViewMode';

export const DividerElementComponentEditMode = memo(({ el }: { el: DividerElement }) => {
  const container = useContainerContext();
  const { id } = el;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex
        id={id}
        className={DIVIDER_CLASS_NAME}
        sx={sx}
        data-orientation={container?.layout === 'column' ? 'horizontal' : 'vertical'}
      />
    </FormElementEditModeWrapper>
  );
});

DividerElementComponentEditMode.displayName = 'DividerElementComponentEditMode';
