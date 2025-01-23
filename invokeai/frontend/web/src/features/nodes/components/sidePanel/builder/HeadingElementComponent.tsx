import { Flex, Heading } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { HEADING_CLASS_NAME, isHeadingElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const LEVEL_TO_SIZE = {
  1: 'xl',
  2: 'lg',
  3: 'md',
  4: 'sm',
  5: 'xs',
} as const;

export const HeadingElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

  if (!el || !isHeadingElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <HeadingElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <HeadingElementComponentEditMode el={el} />;
});

HeadingElementComponent.displayName = 'HeadingElementComponent';

export const HeadingElementComponentViewMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content, level } = data;

  return (
    <Flex id={id} className={HEADING_CLASS_NAME}>
      <Heading size={LEVEL_TO_SIZE[level]}>{content}</Heading>
    </Flex>
  );
});

HeadingElementComponentViewMode.displayName = 'HeadingElementComponentViewMode';

export const HeadingElementComponentEditMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content, level } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <Flex id={id} className={HEADING_CLASS_NAME}>
        <Heading size={LEVEL_TO_SIZE[level]}>{content}</Heading>
      </Flex>
    </FormElementEditModeWrapper>
  );
});

HeadingElementComponentEditMode.displayName = 'HeadingElementComponentEditMode';
