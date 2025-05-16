import { useAppSelector } from 'app/store/storeHooks';

/**
 * Gathers all currently selected model keys from parameters and loras.
 * This includes the main model, VAE, refiner model, controlnet, and loras.
 */
export const useSelectedModelKeys = () => {
  return useAppSelector((state) => {
    const keys = new Set<string>();
    const main = state.params.model;
    const vae = state.params.vae;
    const refiner = state.params.refinerModel;
    const controlnet = state.params.controlLora;
    const loras = state.loras.loras.map((l) => l.model);

    if (main) {
      keys.add(main.key);
    }
    if (vae) {
      keys.add(vae.key);
    }
    if (refiner) {
      keys.add(refiner.key);
    }
    if (controlnet) {
      keys.add(controlnet.key);
    }
    for (const lora of loras) {
      keys.add(lora.key);
    }

    return keys;
  });
};
