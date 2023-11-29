import { WorkflowLibraryContext } from 'features/nodes/components/flow/WorkflowLibrary/context';
import { useContext } from 'react';

export const useWorkflowLibraryContext = () => {
  const context = useContext(WorkflowLibraryContext);
  if (!context) {
    throw new Error(
      'useWorkflowLibraryContext must be used within a WorkflowLibraryContext.Provider'
    );
  }
  return context;
};
