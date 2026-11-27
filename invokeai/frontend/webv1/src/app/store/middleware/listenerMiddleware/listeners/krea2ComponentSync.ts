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
  const availableVaes = [...availableQwenImageVaes, ...availableAnimaVaes];
  const hasSelectedVae = selectedVae !== null && selectedVae !== undefined;
  const hasSelectedEncoder = selectedEncoder !== null && selectedEncoder !== undefined;
  const selectedVaeIsAvailable = hasSelectedVae && availableVaes.some((vae) => vae.key === selectedVae.key);
  const selectedEncoderIsAvailable =
    hasSelectedEncoder && availableEncoders.some((encoder) => encoder.key === selectedEncoder.key);

  return {
    ...(!selectedVaeIsAvailable && (hasSelectedVae || defaultVae) ? { vae: defaultVae ?? null } : {}),
    ...(!selectedEncoderIsAvailable && (hasSelectedEncoder || defaultEncoder)
      ? { encoder: defaultEncoder ?? null }
      : {}),
  };
};
