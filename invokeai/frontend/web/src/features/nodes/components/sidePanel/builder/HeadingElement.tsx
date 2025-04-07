import { useAppSelector } from 'app/store/storeHooks';
import { HeadingElementEditMode } from 'features/nodes/components/sidePanel/builder/HeadingElementEditMode';
import { HeadingElementViewMode } from 'features/nodes/components/sidePanel/builder/HeadingElementViewMode';
import { useElement } from 'features/nodes/components/sidePanel/builder/use-element';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import { isHeadingElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const HeadingElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isHeadingElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <HeadingElementViewMode el={el} />;
  }

  // mode === 'edit'
  return <HeadingElementEditMode el={el} />;
});

HeadingElement.displayName = 'HeadingElement';
