import type { UseDisclosureReturn } from '@invoke-ai/ui';
import { createContext } from 'react';

export const WorkflowLibraryModalContext =
  createContext<UseDisclosureReturn | null>(null);
