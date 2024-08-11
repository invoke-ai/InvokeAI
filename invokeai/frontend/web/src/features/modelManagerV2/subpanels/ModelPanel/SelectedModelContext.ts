import { createContext } from 'react';
import type { AnyModelConfig } from 'services/api/types';

export const SelectedModelContext = createContext<AnyModelConfig | null>(null);
