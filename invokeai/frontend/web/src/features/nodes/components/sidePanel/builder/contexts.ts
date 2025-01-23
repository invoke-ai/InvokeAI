import type { ContainerElement } from 'features/nodes/types/workflow';
import { createContext } from 'react';

export const ContainerContext = createContext<ContainerElement['data'] | null>(null);
export const DepthContext = createContext<number>(0);
