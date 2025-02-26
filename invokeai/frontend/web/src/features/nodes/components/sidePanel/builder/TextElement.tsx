import { useAppSelector } from 'app/store/storeHooks';
import { TextElementEditMode } from 'features/nodes/components/sidePanel/builder/TextElementEditMode';
import { TextElementViewMode } from 'features/nodes/components/sidePanel/builder/TextElementViewMode';
import { selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import { isTextElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const TextElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isTextElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <TextElementViewMode el={el} />;
  }

  // mode === 'edit'
  return <TextElementEditMode el={el} />;
});
TextElement.displayName = 'TextElement';
