import type { UseDisclosureReturn } from '@chakra-ui/react';
import { createContext } from 'react';

export const WorkflowLibraryModalContext =
  createContext<UseDisclosureReturn | null>(null);
