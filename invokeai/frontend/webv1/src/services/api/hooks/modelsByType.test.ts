import type { RootState } from 'app/store/store';
import type * as modelsEndpointModule from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * `buildModelsSelector` filters the installed models with a type guard. Several of those guards take an
 * optional second parameter (`excludeSubmodels`), and `Array#filter` passes the element *index* as the
 * second argument - so a point-free `.filter(typeGuard)` evaluated the first entry with submodels
 * included and every other entry with them excluded. Pool membership then depended on a model's position
 * in the (name-ordered) list rather than on the model.
 */

let configs: AnyModelConfig[] = [];

vi.mock('services/api/endpoints/models', async (importOriginal) => {
  const mod = await importOriginal<typeof modelsEndpointModule>();
  return {
    ...mod,
    selectModelConfigsQuery: () => ({
      data: {
        ids: configs.map((config) => config.key),
        entities: Object.fromEntries(configs.map((config) => [config.key, config])),
      },
    }),
    selectMissingModelsQuery: () => ({ data: undefined }),
  };
});

const { selectAnimaCompatibleVAEModels } = await import('services/api/hooks/modelsByType');

const state = {} as RootState;

const standaloneVae = (key: string, base: string) =>
  ({ key, hash: `${key}-hash`, name: key, base, type: 'vae', format: 'checkpoint' }) as unknown as AnyModelConfig;

const mainWithVaeSubmodel = (key: string, base: string) =>
  ({
    key,
    hash: `${key}-hash`,
    name: key,
    base,
    type: 'main',
    format: 'diffusers',
    submodels: { vae: {} },
  }) as unknown as AnyModelConfig;

beforeEach(() => {
  configs = [];
});

describe('buildModelsSelector', () => {
  it('includes a main-model VAE submodel regardless of its position in the list', () => {
    const first = mainWithVaeSubmodel('flux-main-a', 'flux');
    const second = mainWithVaeSubmodel('flux-main-b', 'flux');
    configs = [first, second];

    expect(selectAnimaCompatibleVAEModels(state).map((config) => config.key)).toEqual([first.key, second.key]);
  });

  it('does not drop entries that only differ from the first by their index', () => {
    configs = [standaloneVae('anima-vae', 'anima'), mainWithVaeSubmodel('flux-main', 'flux')];

    expect(selectAnimaCompatibleVAEModels(state)).toHaveLength(2);
  });

  it('still rejects models the guard does not admit', () => {
    configs = [standaloneVae('sdxl-vae', 'sdxl'), standaloneVae('flux2-vae', 'flux2')];

    expect(selectAnimaCompatibleVAEModels(state)).toEqual([]);
  });
});
