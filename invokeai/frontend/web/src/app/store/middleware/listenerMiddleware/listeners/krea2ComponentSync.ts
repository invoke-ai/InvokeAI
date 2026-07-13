import type { ModelIdentifierField } from 'features/nodes/types/common';

type Krea2ComponentSyncArg = {
  format: string;
  selectedVae: ModelIdentifierField | null;
  selectedEncoder: ModelIdentifierField | null;
  availableQwenImageVaes: ModelIdentifierField[];
  availableAnimaVaes: ModelIdentifierField[];
  availableEncoders: ModelIdentifierField[];
};

type Krea2ComponentUpdates = {
  vae?: ModelIdentifierField | null;
  encoder?: ModelIdentifierField | null;
};

export const getKrea2ComponentUpdates = (arg: Krea2ComponentSyncArg): Krea2ComponentUpdates => {
  const { format, selectedVae, selectedEncoder, availableQwenImageVaes, availableAnimaVaes, availableEncoders } = arg;

  if (format === 'diffusers') {
    return {
      ...(selectedVae ? { vae: null } : {}),
      ...(selectedEncoder ? { encoder: null } : {}),
    };
  }

  const defaultVae = availableQwenImageVaes[0] ?? availableAnimaVaes[0];
  const defaultEncoder = availableEncoders[0];

  return {
    ...(!selectedVae && defaultVae ? { vae: defaultVae } : {}),
    ...(!selectedEncoder && defaultEncoder ? { encoder: defaultEncoder } : {}),
  };
};
