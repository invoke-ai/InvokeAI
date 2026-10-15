import { useAppSelector } from 'app/store/storeHooks';
import { DividerElementEditMode } from 'features/nodes/components/sidePanel/builder/DividerElementEditMode';
import { DividerElementViewMode } from 'features/nodes/components/sidePanel/builder/DividerElementViewMode';
import { useElement } from 'features/nodes/components/sidePanel/builder/use-element';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import { isDividerElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const DividerElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isDividerElement(el)) {
    return;
  }

  if (mode === 'view') {
    return <DividerElementViewMode el={el} />;
  }

  // mode === 'edit'
  return <DividerElementEditMode el={el} />;
});

DividerElement.displayName = 'DividerElement';
