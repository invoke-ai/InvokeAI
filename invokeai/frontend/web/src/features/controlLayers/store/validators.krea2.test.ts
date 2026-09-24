import type { RefImageState } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarningsInContext } from 'features/controlLayers/store/validators';
import type { MainOrExternalModelConfig } from 'services/api/types';
import { describe, expect, it } from 'vitest';

const krea2Model = { base: 'krea-2', type: 'main' } as MainOrExternalModelConfig;
const sdxlModel = { base: 'sdxl', type: 'main' } as MainOrExternalModelConfig;

const image = { original: { image: { image_name: 'a.png' }, width: 64, height: 64 } };

const krea2Entity = (id: string, overrides: Partial<RefImageState> = {}): RefImageState =>
  ({
    id,
    isEnabled: true,
    config: { type: 'krea2_reference_image', styleStrength: 1, image },
    ...overrides,
  }) as RefImageState;

describe('getGlobalReferenceImageWarningsInContext', () => {
  it('does not warn about a single Krea-2 reference image', () => {
    const entities = [krea2Entity('a')];
    expect(getGlobalReferenceImageWarningsInContext(entities[0]!, entities, krea2Model)).toEqual([]);
  });

  it('warns on every Krea-2 reference image after the first', () => {
    // The graph builder consumes exactly one; dropping the rest silently reads as "all are being used".
    const entities = [krea2Entity('a'), krea2Entity('b'), krea2Entity('c')];

    expect(getGlobalReferenceImageWarningsInContext(entities[0]!, entities, krea2Model)).toEqual([]);
    for (const entity of [entities[1]!, entities[2]!]) {
      expect(getGlobalReferenceImageWarningsInContext(entity, entities, krea2Model)).toContain(
        'controlLayers.warnings.krea2OnlyOneReferenceImage'
      );
    }
  });

  it('ignores disabled and image-less entries when deciding which one is used', () => {
    // A disabled entity is not a candidate, so the first *usable* one must stay warning-free.
    const entities = [
      krea2Entity('disabled', { isEnabled: false }),
      krea2Entity('noimage', { config: { type: 'krea2_reference_image', styleStrength: 1, image: null } as never }),
      krea2Entity('used'),
      krea2Entity('extra'),
    ];

    expect(getGlobalReferenceImageWarningsInContext(entities[2]!, entities, krea2Model)).toEqual([]);
    expect(getGlobalReferenceImageWarningsInContext(entities[3]!, entities, krea2Model)).toContain(
      'controlLayers.warnings.krea2OnlyOneReferenceImage'
    );
  });

  it('treats a strength of 0 as disabled when deciding which one is used', () => {
    // 0 is a full bypass - the graph builder skips the entity, so the next one is the one actually used.
    const entities = [
      krea2Entity('bypassed', { config: { type: 'krea2_reference_image', styleStrength: 0, image } as never }),
      krea2Entity('used'),
    ];

    expect(getGlobalReferenceImageWarningsInContext(entities[1]!, entities, krea2Model)).not.toContain(
      'controlLayers.warnings.krea2OnlyOneReferenceImage'
    );
    expect(getGlobalReferenceImageWarningsInContext(entities[0]!, entities, krea2Model)).not.toContain(
      'controlLayers.warnings.krea2OnlyOneReferenceImage'
    );
  });

  it('does not warn when a disabled entity is the extra one', () => {
    const entities = [krea2Entity('used'), krea2Entity('off', { isEnabled: false })];
    expect(getGlobalReferenceImageWarningsInContext(entities[1]!, entities, krea2Model)).not.toContain(
      'controlLayers.warnings.krea2OnlyOneReferenceImage'
    );
  });

  it('leaves other bases alone', () => {
    // SDXL chains multiple IP-Adapters, so several reference images are legitimate there.
    const entities = [krea2Entity('a'), krea2Entity('b')];
    expect(getGlobalReferenceImageWarningsInContext(entities[1]!, entities, sdxlModel)).not.toContain(
      'controlLayers.warnings.krea2OnlyOneReferenceImage'
    );
  });
});
