import { createContext } from 'react';

type VoidFunc = () => void;

type ImageUploaderTriggerContextType = VoidFunc | null;

export const ImageUploaderTriggerContext =
  createContext<ImageUploaderTriggerContextType>(null);
