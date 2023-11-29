import { UseDisclosureReturn } from '@chakra-ui/react';
import { createContext } from 'react';

export const WorkflowLibraryContext = createContext<UseDisclosureReturn | null>(
  null
);
