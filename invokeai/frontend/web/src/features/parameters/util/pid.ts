import type { BaseModelType } from 'features/nodes/types/common';

/**
 * Maps a main-model base to the PiD decoder base whose checkpoints are valid for it.
 *
 * PiD decoders are trained per backbone, so only a base-matching decoder may be used (e.g. a FLUX.2 decoder for a
 * FLUX.2 main model). Returns `null` for bases whose graph builder does not (yet) wire a PiD decode - PiD is
 * unsupported for those. Additional bases are added here as their graph builders gain PiD support.
 */
export const getPidDecoderBaseForMainBase = (base?: BaseModelType | null): BaseModelType | null => {
  switch (base) {
    case 'flux':
    case 'flux2':
    case 'sd-3':
    case 'sdxl':
      return base;
    default:
      return null;
  }
};

/** Whether the given main-model base supports PiD decoding (i.e. its graph builder wires a PiD decode). */
export const getIsPidSupportedBase = (base?: BaseModelType | null): boolean =>
  getPidDecoderBaseForMainBase(base) !== null;
