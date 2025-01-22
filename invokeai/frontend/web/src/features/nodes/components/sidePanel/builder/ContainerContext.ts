import { createContext, useContext } from 'react';

export const ContainerDirectionContext = createContext<'row' | 'column' | null>(null);

export const useContainerDirectionContext = () => {
  const containerDirection = useContext(ContainerDirectionContext);
  return containerDirection;
};
