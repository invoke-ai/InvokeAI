import type { ContainerElement } from 'features/nodes/types/workflow';
import { createContext, useContext } from 'react';

export const ContainerContext = createContext<ContainerElement['data'] | null>(null);

export const useContainerContext = () => {
  const containerDirection = useContext(ContainerContext);
  return containerDirection;
};
