import { WorkflowLibraryModalContext } from 'features/workflowLibrary/context/WorkflowLibraryModalContext';
import { useContext } from 'react';

export const useWorkflowLibraryModalContext = () => {
  const context = useContext(WorkflowLibraryModalContext);
  if (!context) {
    throw new Error('useWorkflowLibraryContext must be used within a WorkflowLibraryContext.Provider');
  }
  return context;
};
